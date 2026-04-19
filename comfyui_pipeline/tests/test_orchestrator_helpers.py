"""Unit tests for orchestrator helpers that don't require a live ComfyUI."""
from __future__ import annotations

import json
import subprocess
from unittest.mock import patch

from comfyui_pipeline.src import orchestrator as orch
from comfyui_pipeline.src.orchestrator import (
    PipelineConfig,
    PipelineResult,
    ScenePipeline,
    SceneArtifacts,
)
from comfyui_pipeline.src.schema import Scenario, Scene


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


def test_build_image_workflow_uses_ksampler_advanced(tmp_path) -> None:
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
    sampler_id = orch._find_node_by_title(wf, "Sample")
    sampler = wf[sampler_id]
    assert sampler["class_type"] == "KSamplerAdvanced"
    assert sampler["inputs"]["noise_seed"] == 1234
    assert sampler["inputs"]["add_noise"] == "enable"
    assert "seed" not in sampler["inputs"]


def test_build_video_workflow_wires_ltx23_nodes(tmp_path) -> None:
    cfg = PipelineConfig(
        output_dir=tmp_path,
        ltx_checkpoint="ltx23.safetensors",
        ltx_lora="ltx23-lora.safetensors",
        ltx_lora_strength=0.7,
        ltx_text_encoder="gemma3-12b.safetensors",
        video_fps=25.0,
    )
    pipeline = ScenePipeline(cfg)
    scene = Scene(
        id=1,
        description="opening",
        image_prompt="a lighthouse at dawn",
        video_prompt="waves crash, camera pans left",
        duration_seconds=3.0,
    )
    wf = pipeline._build_video_workflow(scene, "uploads/keyframe.png", "cinematic", seed=42)

    # Checkpoint, encoder, LoRA wiring
    assert wf[orch._find_node_by_title(wf, "Load LTX checkpoint")]["inputs"]["ckpt_name"] == "ltx23.safetensors"
    enc = wf[orch._find_node_by_title(wf, "LTX text encoder")]
    assert enc["class_type"] == "LTXAVTextEncoderLoader"
    assert enc["inputs"]["text_encoder"] == "gemma3-12b.safetensors"
    assert enc["inputs"]["ckpt_name"] == "ltx23.safetensors"
    lora = wf[orch._find_node_by_title(wf, "Apply LTX distilled LoRA")]
    assert lora["class_type"] == "LoraLoaderModelOnly"
    assert lora["inputs"]["lora_name"] == "ltx23-lora.safetensors"
    assert lora["inputs"]["strength_model"] == 0.7

    # Image load + latent dims
    assert wf[orch._find_node_by_title(wf, "Load keyframe image")]["inputs"]["image"] == "uploads/keyframe.png"
    latent = wf[orch._find_node_by_title(wf, "LTX empty latent")]["inputs"]
    assert latent["length"] == pipeline._frames_for_duration(3.0)

    # Sampler stack
    sampler_adv = wf[orch._find_node_by_title(wf, "Sample video latent")]
    assert sampler_adv["class_type"] == "SamplerCustomAdvanced"
    assert wf[orch._find_node_by_title(wf, "Random noise")]["inputs"]["noise_seed"] == 42
    assert wf[orch._find_node_by_title(wf, "CFG guider")]["inputs"]["cfg"] == cfg.video_cfg


def test_build_script_workflow_round_trips(tmp_path) -> None:
    cfg = PipelineConfig(output_dir=tmp_path, qwen_model="Qwen/Qwen3-8B")
    pipeline = ScenePipeline(cfg)
    wf = pipeline._build_script_workflow("a cat surfs", "you are a director", seed=7)
    qwen = wf[orch._find_node_by_title(wf, "Qwen scenario generator")]["inputs"]
    assert qwen["model_name_or_path"] == "Qwen/Qwen3-8B"
    assert qwen["seed"] == 7
    # The workflow must still be JSON-serialisable after patching.
    json.dumps(wf)


def test_concat_final_video_invokes_ffmpeg(tmp_path, monkeypatch) -> None:
    cfg = PipelineConfig(output_dir=tmp_path, ffmpeg_binary="ffmpeg-fake")
    pipeline = ScenePipeline(cfg)

    # Two fake scene mp4s side-by-side.
    video_1 = tmp_path / "scene_01.mp4"
    video_2 = tmp_path / "scene_02.mp4"
    video_1.write_bytes(b"fake1")
    video_2.write_bytes(b"fake2")
    scenario = Scenario(
        title="t", style="s",
        scenes=[
            Scene(id=1, description="a", image_prompt="i", video_prompt="v", duration_seconds=1.0),
            Scene(id=2, description="b", image_prompt="i", video_prompt="v", duration_seconds=1.0),
        ],
    )
    result = PipelineResult(scenario=scenario, output_dir=tmp_path)
    result.scene_artifacts = [
        SceneArtifacts(scene=scenario.scenes[0], image_path=tmp_path / "a.png", video_path=video_1),
        SceneArtifacts(scene=scenario.scenes[1], image_path=tmp_path / "b.png", video_path=video_2),
    ]

    calls: list[list[str]] = []

    def fake_run(cmd, capture_output=True, text=True):
        calls.append(list(cmd))
        # Create the output file so the caller sees success.
        out_path = cmd[-1]
        from pathlib import Path as _P
        _P(out_path).write_bytes(b"final")
        class _R:
            returncode = 0
            stdout = ""
            stderr = ""
        return _R()

    monkeypatch.setattr(
        "comfyui_pipeline.src.orchestrator.shutil.which",
        lambda _name: "/usr/bin/ffmpeg-fake",
    )
    with patch.object(subprocess, "run", side_effect=fake_run):
        final = pipeline._concat_final_video(result)

    assert final == tmp_path / "final.mp4"
    assert final.exists()
    assert calls and calls[0][0] == "ffmpeg-fake"
    # concat_list.txt was created with both scenes.
    lst = (tmp_path / "concat_list.txt").read_text()
    assert str(video_1.resolve()) in lst
    assert str(video_2.resolve()) in lst
