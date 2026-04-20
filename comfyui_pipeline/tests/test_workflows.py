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


def test_scene_video_workflow_is_ltx_2_3_av() -> None:
    data = json.loads((WORKFLOW_DIR / "scene_video_api.json").read_text(encoding="utf-8"))
    class_types = {node["class_type"] for node in data.values()}
    # LTX-2.3 is an audio-visual model: the sampler consumes a unified
    # AV-latent (video + audio) and emits one back. The I2V workflow must
    # therefore fan out into two branches (EmptyLTXVLatentVideo +
    # LTXVImgToVideoConditionOnly on the video side, LTXVAudioVAELoader +
    # LTXVEmptyLatentAudio on the audio side) merged by LTXVConcatAVLatent,
    # and split again after sampling by LTXVSeparateAVLatent. The pure
    # video-only LTXVImgToVideo node from older LTX-2.0 workflows is NOT
    # compatible and produces 4-D vs 3-D tensor shape mismatches.
    for expected in (
        "LTXAVTextEncoderLoader",
        "LTXVImgToVideoConditionOnly",
        "EmptyLTXVLatentVideo",
        "LTXVAudioVAELoader",
        "LTXVEmptyLatentAudio",
        "LTXVConcatAVLatent",
        "LTXVSeparateAVLatent",
        "LTXVConditioning",
        "LTXVScheduler",
        "LTXVTiledVAEDecode",
        "KSamplerSelect",
        "RandomNoise",
        "CFGGuider",
        "SamplerCustomAdvanced",
        "CreateVideo",
        "SaveVideo",
    ):
        assert expected in class_types, f"scene_video_api.json must contain {expected}"
    # Reject older topologies that don't work with LTX-2.3 AV model.
    assert "SamplerCustom" not in class_types
    assert "LTXVImgToVideo" not in class_types, (
        "LTX-2.3 is AV — use LTXVImgToVideoConditionOnly + LTXVConcatAVLatent instead"
    )
    assert "CLIPLoader" not in class_types, (
        "LTX-2.3 needs LTXAVTextEncoderLoader (encoder+ckpt pair), not CLIPLoader"
    )


def test_scene_video_av_text_encoder_pairs_encoder_with_ckpt() -> None:
    """LTXAVTextEncoderLoader requires BOTH the Gemma encoder file AND the
    LTX checkpoint (it reads the cross-attention projection weights that
    pair Gemma embeddings with the video model)."""
    data = json.loads((WORKFLOW_DIR / "scene_video_api.json").read_text(encoding="utf-8"))
    enc_nodes = [n for n in data.values() if n["class_type"] == "LTXAVTextEncoderLoader"]
    assert len(enc_nodes) == 1, "scene_video must have exactly one LTXAVTextEncoderLoader"
    enc = enc_nodes[0]
    assert enc["inputs"].get("text_encoder"), "LTXAVTextEncoderLoader needs text_encoder"
    assert enc["inputs"].get("ckpt_name"), "LTXAVTextEncoderLoader needs ckpt_name"


def test_scene_video_av_latent_fanout_fanin() -> None:
    """Validate that the AV-latent is the single latent fed into the sampler
    (topology bug guard): LTXVConcatAVLatent output must feed both the
    scheduler latent input and the sampler latent_image input."""
    data = json.loads((WORKFLOW_DIR / "scene_video_api.json").read_text(encoding="utf-8"))
    concat_id = next(
        nid for nid, n in data.items() if n["class_type"] == "LTXVConcatAVLatent"
    )
    sched_node = next(n for n in data.values() if n["class_type"] == "LTXVScheduler")
    sampler_node = next(
        n for n in data.values() if n["class_type"] == "SamplerCustomAdvanced"
    )
    assert sched_node["inputs"].get("latent") == [concat_id, 0]
    assert sampler_node["inputs"].get("latent_image") == [concat_id, 0]
    sep_node = next(
        n for n in data.values() if n["class_type"] == "LTXVSeparateAVLatent"
    )
    sampler_id = next(
        nid for nid, n in data.items() if n["class_type"] == "SamplerCustomAdvanced"
    )
    assert sep_node["inputs"].get("av_latent") == [sampler_id, 0]
