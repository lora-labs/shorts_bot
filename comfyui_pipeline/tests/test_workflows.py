"""Structural checks that the shipped workflows are valid ComfyUI API JSON."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

WORKFLOW_DIR = Path(__file__).resolve().parent.parent / "workflows"


@pytest.mark.parametrize(
    "filename",
    [
        "script_gen_api.json",
        "scene_image_api.json",
        "scene_video_api.json",
    ],
)
def test_workflow_shape(filename: str) -> None:
    path = WORKFLOW_DIR / filename
    assert path.is_file(), f"missing workflow: {path}"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict) and data, "workflow must be a non-empty dict"
    for node_id, node in data.items():
        assert isinstance(node_id, str) and node_id.isdigit(), f"node id {node_id!r} must be numeric string"
        assert "class_type" in node, f"node {node_id} missing class_type"
        assert isinstance(node.get("inputs"), dict), f"node {node_id} missing inputs dict"
        meta = node.get("_meta") or {}
        assert isinstance(meta, dict)
        assert meta.get("title"), f"node {node_id} missing _meta.title (required by orchestrator)"


def test_scene_video_links_resolve() -> None:
    data = json.loads((WORKFLOW_DIR / "scene_video_api.json").read_text(encoding="utf-8"))
    ids = set(data.keys())
    for node_id, node in data.items():
        for input_name, value in node["inputs"].items():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                src_id, _ = value
                assert src_id in ids, (
                    f"node {node_id}.{input_name} references missing source node {src_id}"
                )


def test_script_workflow_has_qwen_and_save_nodes() -> None:
    data = json.loads((WORKFLOW_DIR / "script_gen_api.json").read_text(encoding="utf-8"))
    class_types = {node["class_type"] for node in data.values()}
    assert "QwenScenarioGenerator" in class_types
    assert "SaveTextToFile" in class_types


def test_scene_image_workflow_uses_ksampler_advanced() -> None:
    data = json.loads((WORKFLOW_DIR / "scene_image_api.json").read_text(encoding="utf-8"))
    class_types = {node["class_type"] for node in data.values()}
    assert "KSamplerAdvanced" in class_types
    assert "KSampler" not in class_types, (
        "scene_image_api.json must use KSamplerAdvanced, not the basic KSampler"
    )


def test_scene_video_workflow_is_ltx_2_3() -> None:
    data = json.loads((WORKFLOW_DIR / "scene_video_api.json").read_text(encoding="utf-8"))
    class_types = {node["class_type"] for node in data.values()}
    # LTX-2.3 single-stage I2V chain. As of ComfyUI core 0.18+ the builtin
    # LTXVImgToVideo node subsumes both the old EmptyLTXVLatentVideo and
    # LTXVImgToVideoConditionOnly nodes (it accepts positive/negative/vae/
    # image and width/height/length and emits positive/negative/latent in
    # one shot). The old names were removed from both ComfyUI core and the
    # Lightricks ComfyUI-LTXVideo extension, so we must not reference them.
    for expected in (
        "LTXAVTextEncoderLoader",
        "LTXVConditioning",
        "LTXVImgToVideo",
        "LTXVScheduler",
        "KSamplerSelect",
        "RandomNoise",
        "CFGGuider",
        "SamplerCustomAdvanced",
        "CreateVideo",
        "SaveVideo",
    ):
        assert expected in class_types, f"scene_video_api.json must contain {expected}"
    # Old LTX-2.0 / removed LTX-2.3 pre-release nodes must be gone.
    assert "SamplerCustom" not in class_types
    assert "EmptyLTXVLatentVideo" not in class_types
    assert "LTXVImgToVideoConditionOnly" not in class_types
