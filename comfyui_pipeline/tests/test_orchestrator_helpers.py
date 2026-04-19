"""Unit tests for orchestrator helpers that don't require a live ComfyUI."""
from __future__ import annotations

import json

from comfyui_pipeline.src import orchestrator as orch
from comfyui_pipeline.src.orchestrator import PipelineConfig, ScenePipeline
from comfyui_pipeline.src.schema import Scene


def test_strip_json_fence_removes_markdown() -> None:
    assert orch._strip_json_fence("```json\n{\"a\":1}\n```") == '{"a":1}'
    assert orch._strip_json_fence("```\n{\"a\":1}\n```") == '{"a":1}'
    assert orch._strip_json_fence("  {\"a\":1}  ") == '{"a":1}'


def test_find_node_by_title() -> None:
    wf = {
        "1": {"class_type": "X", "inputs": {}, "_meta": {"title": "Alpha"}},
        "2": {"class_type": "Y", "inputs": {}, "_meta": {"title": "Beta"}},
    }
    assert orch._find_node_by_title(wf, "Beta") == "2"


def test_frames_for_duration_is_valid_ltx_length(tmp_path) -> None:
    cfg = PipelineConfig(output_dir=tmp_path, video_fps=25.0)
    pipeline = ScenePipeline(cfg)
    for seconds in (0.1, 1.0, 3.0, 4.0, 7.5, 10.0):
        n = pipeline._frames_for_duration(seconds)
        assert n >= 9
        assert (n - 1) % 8 == 0


def test_build_image_workflow_patches_prompts(tmp_path) -> None:
    cfg = PipelineConfig(output_dir=tmp_path, sdxl_checkpoint="custom_sdxl.safetensors")
    pipeline = ScenePipeline(cfg)
    scene = Scene(
        id=3,
        description="hero walks",
        image_prompt="hero walking, cinematic",
        video_prompt="slow walk, dolly in",
        duration_seconds=4,
    )
    wf = pipeline._build_image_workflow(scene, "warm palette", seed=1234)
    assert wf[orch._find_node_by_title(wf, "Load SDXL checkpoint")]["inputs"]["ckpt_name"] == "custom_sdxl.safetensors"
    pos = wf[orch._find_node_by_title(wf, "Positive prompt")]["inputs"]["text"]
    assert "warm palette" in pos and "hero walking" in pos
    save = wf[orch._find_node_by_title(wf, "Save scene image")]["inputs"]["filename_prefix"]
    assert save == "scene_03_image"
    sampler = wf[orch._find_node_by_title(wf, "Sample")]["inputs"]
    assert sampler["seed"] == 1234


def test_build_video_workflow_wires_image_and_length(tmp_path) -> None:
    cfg = PipelineConfig(output_dir=tmp_path, ltx_checkpoint="ltx.safetensors", video_fps=25.0)
    pipeline = ScenePipeline(cfg)
    scene = Scene(
        id=1,
        description="opening",
        image_prompt="a lighthouse at dawn",
        video_prompt="waves crash, camera pans left",
        duration_seconds=3.0,
    )
    wf = pipeline._build_video_workflow(scene, "uploads/keyframe.png", "cinematic", seed=42)
    assert wf[orch._find_node_by_title(wf, "Load keyframe image")]["inputs"]["image"] == "uploads/keyframe.png"
    assert wf[orch._find_node_by_title(wf, "Load LTX checkpoint")]["inputs"]["ckpt_name"] == "ltx.safetensors"
    i2v = wf[orch._find_node_by_title(wf, "LTX img-to-video")]["inputs"]
    assert i2v["length"] == pipeline._frames_for_duration(3.0)
    sampler = wf[orch._find_node_by_title(wf, "Sample video latent")]["inputs"]
    assert sampler["noise_seed"] == 42


def test_build_script_workflow_round_trips(tmp_path) -> None:
    cfg = PipelineConfig(output_dir=tmp_path, qwen_model="Qwen/Qwen2.5-3B-Instruct")
    pipeline = ScenePipeline(cfg)
    wf = pipeline._build_script_workflow("a cat surfs", "you are a director", seed=7)
    qwen = wf[orch._find_node_by_title(wf, "Qwen scenario generator")]["inputs"]
    assert qwen["model_name_or_path"] == "Qwen/Qwen2.5-3B-Instruct"
    assert qwen["seed"] == 7
    # The workflow must still be JSON-serialisable after patching.
    json.dumps(wf)
