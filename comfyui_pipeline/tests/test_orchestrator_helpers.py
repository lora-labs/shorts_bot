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


def test_image_prompt_prepends_character_sheet(tmp_path) -> None:
    """The character_sheet is the single source of truth for the subject's
    appearance. It must be prepended (so tokens land near the front of the
    CLIP context, where SDXL weights them highest) to every scene's prompt."""
    cfg = PipelineConfig(output_dir=tmp_path)
    pipeline = ScenePipeline(cfg)
    scene = Scene(
        id=1,
        description="opens door",
        image_prompt="in a cozy kitchen at golden hour, medium shot",
        video_prompt="hand opens door slowly",
        duration_seconds=3.0,
    )
    sheet = "fluffy white cat with one blue eye and one green eye, red bowtie"
    wf = pipeline._build_image_workflow(scene, "cinematic", seed=1, character_sheet=sheet)
    pos = wf[orch._find_node_by_title(wf, "Positive prompt")]["inputs"]["text"]
    # character_sheet must appear BEFORE the scene's image_prompt.
    assert pos.index(sheet) < pos.index("cozy kitchen")
    assert "cinematic" in pos


def test_video_prompt_prepends_character_sheet(tmp_path) -> None:
    cfg = PipelineConfig(output_dir=tmp_path)
    pipeline = ScenePipeline(cfg)
    scene = Scene(
        id=2,
        description="rides skateboard",
        image_prompt="on a sunny street",
        video_prompt="skateboard rolls forward, camera follows",
        duration_seconds=3.0,
    )
    sheet = "orange tabby cat wearing red helmet"
    wf = pipeline._build_video_workflow(
        scene, "keyframe.png", "cinematic", seed=1, character_sheet=sheet
    )
    vid_pos = wf[orch._find_node_by_title(wf, "Video positive prompt")]["inputs"]["text"]
    assert vid_pos.index(sheet) < vid_pos.index("skateboard rolls")


def test_default_config_is_vertical_9_16(tmp_path) -> None:
    """Output targets TikTok / Reels / Shorts — vertical 9:16 by default."""
    cfg = PipelineConfig(output_dir=tmp_path)
    assert cfg.image_height > cfg.image_width
    assert cfg.video_height > cfg.video_width
    # Both SDXL and LTX need dims that respect their bucket/stride rules.
    assert cfg.image_width % 64 == 0 and cfg.image_height % 64 == 0
    assert cfg.video_width % 32 == 0 and cfg.video_height % 32 == 0


def test_image_workflow_wires_9_16_latent_dims(tmp_path) -> None:
    cfg = PipelineConfig(output_dir=tmp_path, image_width=768, image_height=1344)
    pipeline = ScenePipeline(cfg)
    scene = Scene(
        id=1,
        description="x",
        image_prompt="a lighthouse",
        video_prompt="x",
        duration_seconds=3.0,
    )
    wf = pipeline._build_image_workflow(scene, "cinematic", seed=1)
    latent = wf[orch._find_node_by_title(wf, "Empty latent 9:16")]["inputs"]
    assert latent["width"] == 768
    assert latent["height"] == 1344


def test_scene1_uses_plain_workflow_no_ip_adapter(tmp_path) -> None:
    """Scene 1 has no reference image yet — must fall back to the plain
    SDXL workflow without IPAdapter nodes."""
    cfg = PipelineConfig(output_dir=tmp_path)
    pipeline = ScenePipeline(cfg)
    scene = Scene(
        id=1,
        description="opens door",
        image_prompt="in a cozy kitchen",
        video_prompt="x",
        duration_seconds=3.0,
    )
    wf = pipeline._build_image_workflow(scene, "cinematic", seed=1)
    class_types = {n["class_type"] for n in wf.values()}
    assert "IPAdapterAdvanced" not in class_types


def test_scene2plus_uses_ipa_workflow_when_reference_provided(tmp_path) -> None:
    """When a reference_image name is passed, orchestrator must switch to
    the IPA workflow and wire the reference through LoadImage →
    PrepImageForClipVision → IPAdapterAdvanced → KSamplerAdvanced.model."""
    cfg = PipelineConfig(
        output_dir=tmp_path,
        ip_adapter_model="ip-adapter-plus_sdxl_vit-h.safetensors",
        clip_vision_model="CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors",
        ip_adapter_weight=0.55,
        ip_adapter_weight_type="style transfer",
    )
    pipeline = ScenePipeline(cfg)
    scene = Scene(
        id=2,
        description="sits by window",
        image_prompt="on a windowsill at night",
        video_prompt="x",
        duration_seconds=3.0,
    )
    wf = pipeline._build_image_workflow(
        scene,
        "cinematic",
        seed=1,
        character_sheet="fluffy white cat",
        reference_image="scene_01_abc.png",
    )
    class_types = {n["class_type"] for n in wf.values()}
    assert "IPAdapterAdvanced" in class_types
    assert "IPAdapterModelLoader" in class_types
    assert "CLIPVisionLoader" in class_types
    assert "PrepImageForClipVision" in class_types

    ipa_id = orch._find_node_by_title(wf, "Apply IP-Adapter")
    ipa = wf[ipa_id]
    assert ipa["inputs"]["weight"] == 0.55
    assert ipa["inputs"]["weight_type"] == "style transfer"
    # model input of KSamplerAdvanced must be the IPAdapter output (id 12),
    # not the raw checkpoint (id 1).
    sampler = wf[orch._find_node_by_title(wf, "Sample")]
    assert sampler["inputs"]["model"] == [ipa_id, 0]

    assert wf[orch._find_node_by_title(wf, "Load IP-Adapter")]["inputs"]["ipadapter_file"] == (
        "ip-adapter-plus_sdxl_vit-h.safetensors"
    )
    assert wf[orch._find_node_by_title(wf, "Load CLIP Vision")]["inputs"]["clip_name"] == (
        "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"
    )
    assert wf[orch._find_node_by_title(wf, "Load reference image")]["inputs"]["image"] == (
        "scene_01_abc.png"
    )
    # Character sheet must still prepend even in the IPA variant.
    pos = wf[orch._find_node_by_title(wf, "Positive prompt")]["inputs"]["text"]
    assert pos.index("fluffy white cat") < pos.index("on a windowsill")


def test_ipa_disabled_by_config_falls_back_to_plain_workflow(tmp_path) -> None:
    """If the user sets use_ip_adapter=False, the orchestrator must NEVER
    pick the IPA workflow even when a reference_image is supplied —
    useful when the IPAdapter model isn't installed on this box."""
    cfg = PipelineConfig(output_dir=tmp_path, use_ip_adapter=False)
    pipeline = ScenePipeline(cfg)
    scene = Scene(
        id=3,
        description="walks",
        image_prompt="in a park",
        video_prompt="x",
        duration_seconds=3.0,
    )
    wf = pipeline._build_image_workflow(
        scene, "cinematic", seed=1, reference_image="scene_01.png"
    )
    class_types = {n["class_type"] for n in wf.values()}
    assert "IPAdapterAdvanced" not in class_types


def test_prompt_composition_handles_empty_character_sheet(tmp_path) -> None:
    """Backwards compat: scenarios without a character_sheet (e.g. from an
    older Qwen prompt template) must still produce a valid prompt."""
    cfg = PipelineConfig(output_dir=tmp_path)
    pipeline = ScenePipeline(cfg)
    scene = Scene(
        id=1,
        description="x",
        image_prompt="a lonely lighthouse",
        video_prompt="waves crash",
        duration_seconds=3.0,
    )
    prompt = pipeline._compose_image_prompt(scene, "cinematic", "")
    assert prompt.startswith("a lonely lighthouse")
    assert "cinematic" in prompt


def test_build_video_workflow_wires_ltx23_av_nodes(tmp_path) -> None:
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
    # LTXAVTextEncoderLoader requires BOTH the Gemma encoder file and the
    # LTX checkpoint (it reads the cross-attention projection weights that
    # pair Gemma embeddings with the video model).
    enc = wf[orch._find_node_by_title(wf, "LTX text encoder")]
    assert enc["class_type"] == "LTXAVTextEncoderLoader"
    assert enc["inputs"]["text_encoder"] == "gemma3-12b.safetensors"
    assert enc["inputs"]["ckpt_name"] == "ltx23.safetensors"
    # Audio VAE loader re-reads the same LTX checkpoint.
    audio_vae = wf[orch._find_node_by_title(wf, "Load audio VAE")]
    assert audio_vae["class_type"] == "LTXVAudioVAELoader"
    assert audio_vae["inputs"]["ckpt_name"] == "ltx23.safetensors"
    lora = wf[orch._find_node_by_title(wf, "Apply LTX distilled LoRA")]
    assert lora["class_type"] == "LoraLoaderModelOnly"
    assert lora["inputs"]["lora_name"] == "ltx23-lora.safetensors"
    assert lora["inputs"]["strength_model"] == 0.7

    # Image load + latent dims. LTX-2.3 AV: width/height/length live on
    # EmptyLTXVLatentVideo; LTXVImgToVideoConditionOnly takes the image
    # and the empty latent and emits a conditioned video latent.
    assert wf[orch._find_node_by_title(wf, "Load keyframe image")]["inputs"]["image"] == "uploads/keyframe.png"
    video_latent = wf[orch._find_node_by_title(wf, "Empty video latent")]
    assert video_latent["class_type"] == "EmptyLTXVLatentVideo"
    length = pipeline._frames_for_duration(3.0)
    assert video_latent["inputs"]["length"] == length
    assert video_latent["inputs"]["width"] == cfg.video_width
    assert video_latent["inputs"]["height"] == cfg.video_height
    i2v = wf[orch._find_node_by_title(wf, "LTX image to video condition")]
    assert i2v["class_type"] == "LTXVImgToVideoConditionOnly"
    # Audio branch matches video length + fps (AV sampler enforces this).
    audio_latent = wf[orch._find_node_by_title(wf, "Empty audio latent")]
    assert audio_latent["class_type"] == "LTXVEmptyLatentAudio"
    assert audio_latent["inputs"]["frames_number"] == length
    assert audio_latent["inputs"]["frame_rate"] == int(round(cfg.video_fps))

    # Sampler stack consumes the concatenated AV-latent.
    sampler_adv = wf[orch._find_node_by_title(wf, "Sample AV latent")]
    assert sampler_adv["class_type"] == "SamplerCustomAdvanced"
    assert wf[orch._find_node_by_title(wf, "Random noise")]["inputs"]["noise_seed"] == 42
    assert wf[orch._find_node_by_title(wf, "CFG guider")]["inputs"]["cfg"] == cfg.video_cfg


def test_build_script_workflow_round_trips(tmp_path) -> None:
    cfg = PipelineConfig(
        output_dir=tmp_path,
        qwen_model="Qwen/Qwen3-8B",
        qwen_device="cpu",
        qwen_keep_loaded=False,
    )
    pipeline = ScenePipeline(cfg)
    wf = pipeline._build_script_workflow("a cat surfs", "you are a director", seed=7)
    qwen = wf[orch._find_node_by_title(wf, "Qwen scenario generator")]["inputs"]
    assert qwen["model_name_or_path"] == "Qwen/Qwen3-8B"
    assert qwen["seed"] == 7
    assert qwen["device"] == "cpu"
    assert qwen["keep_loaded"] is False
    # The workflow must still be JSON-serialisable after patching.
    json.dumps(wf)


def test_qwen_node_forces_local_files_only_for_local_paths(tmp_path) -> None:
    """If ``model_name_or_path`` points at a real on-disk directory, the
    Qwen node must pass ``local_files_only=True`` to transformers so it
    never tries to contact huggingface.co (which re-downloads shards if
    the local snapshot doesn't match the exact revision HF expects, or
    stalls on xet-token requests behind a VPN).

    For a bare HF repo id (``Qwen/Qwen3-8B``), ``local_files_only`` must
    stay False so a fresh install can still download the model."""
    import sys
    from pathlib import Path

    # Tests run from repo root; make sure the custom_nodes dir is on sys.path
    # even if conftest hasn't done it.
    custom_nodes = Path(__file__).parent.parent / "custom_nodes" / "scene_pipeline"
    if str(custom_nodes) not in sys.path:
        sys.path.insert(0, str(custom_nodes))

    import qwen_node

    # Non-existent path / bare HF id → not treated as local.
    assert qwen_node._looks_like_local_path("Qwen/Qwen3-8B") is False
    assert qwen_node._looks_like_local_path("not/a/real/path") is False

    # Real on-disk dir → treated as local.
    local_dir = tmp_path / "Qwen3-8B"
    local_dir.mkdir()
    assert qwen_node._looks_like_local_path(str(local_dir)) is True


def test_pipeline_config_defaults_are_low_vram_friendly() -> None:
    """Defaults should work on 12 GB GPUs out of the box: Qwen splits across
    CPU+GPU via accelerate and unloads itself before SDXL/LTX run."""
    cfg = PipelineConfig()
    assert cfg.qwen_device == "auto"
    assert cfg.qwen_keep_loaded is False


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


def test_resolve_sdxl_checkpoint_returns_default_for_auto(tmp_path) -> None:
    cfg = PipelineConfig(output_dir=tmp_path, sdxl_checkpoint="default.safetensors")
    pipeline = ScenePipeline(cfg)
    assert pipeline._resolve_sdxl_checkpoint("auto") == "default.safetensors"
    assert pipeline._resolve_sdxl_checkpoint(None) == "default.safetensors"
    assert pipeline._resolve_sdxl_checkpoint("") == "default.safetensors"


def test_resolve_sdxl_checkpoint_maps_preset_to_file(tmp_path) -> None:
    cfg = PipelineConfig(
        output_dir=tmp_path,
        sdxl_checkpoint="default.safetensors",
        checkpoint_presets={
            "cinematic_photo": "jugg.safetensors",
            "anime": "animagine.safetensors",
        },
    )
    pipeline = ScenePipeline(cfg)
    assert pipeline._resolve_sdxl_checkpoint("cinematic_photo") == "jugg.safetensors"
    assert pipeline._resolve_sdxl_checkpoint("anime") == "animagine.safetensors"


def test_resolve_sdxl_checkpoint_falls_back_when_preset_missing_from_map(tmp_path) -> None:
    cfg = PipelineConfig(
        output_dir=tmp_path,
        sdxl_checkpoint="default.safetensors",
        checkpoint_presets={"anime": "animagine.safetensors"},
    )
    pipeline = ScenePipeline(cfg)
    # 'photoreal' not in user's map → fall back to default, don't crash.
    assert pipeline._resolve_sdxl_checkpoint("photoreal") == "default.safetensors"


def test_resolve_sdxl_checkpoint_override_normalises_casing_and_separators(tmp_path) -> None:
    """User-facing labels ('Cinematic Photo', 'cinematic-photo') must be
    accepted so Gradio dropdowns and Telegram commands can push any
    reasonable variant without silently degrading to the default."""
    cfg = PipelineConfig(
        output_dir=tmp_path,
        sdxl_checkpoint="default.safetensors",
        checkpoint_presets={"cinematic_photo": "jugg.safetensors"},
    )
    pipeline = ScenePipeline(cfg)
    for sloppy in ("Cinematic Photo", "cinematic-photo", "  CINEMATIC_PHOTO  "):
        pipeline.config.style_preset_override = sloppy
        assert pipeline._resolve_sdxl_checkpoint("auto") == "jugg.safetensors", sloppy


def test_resolve_sdxl_checkpoint_override_wins_over_scenario(tmp_path) -> None:
    cfg = PipelineConfig(
        output_dir=tmp_path,
        sdxl_checkpoint="default.safetensors",
        checkpoint_presets={
            "anime": "animagine.safetensors",
            "cinematic_photo": "jugg.safetensors",
        },
        style_preset_override="anime",
    )
    pipeline = ScenePipeline(cfg)
    assert pipeline._resolve_sdxl_checkpoint("cinematic_photo") == "animagine.safetensors"


def test_resolve_sdxl_checkpoint_degrades_when_file_missing_on_disk(tmp_path) -> None:
    """With ``sdxl_checkpoints_dir`` configured, a preset pointing at a
    missing file must fall back to the default — a broken download
    should not kill the whole run."""
    cfg = PipelineConfig(
        output_dir=tmp_path,
        sdxl_checkpoint="default.safetensors",
        checkpoint_presets={"anime": "animagine.safetensors"},
        sdxl_checkpoints_dir=str(tmp_path),
    )
    pipeline = ScenePipeline(cfg)
    assert pipeline._resolve_sdxl_checkpoint("anime") == "default.safetensors"
    (tmp_path / "animagine.safetensors").write_bytes(b"fake")
    assert pipeline._resolve_sdxl_checkpoint("anime") == "animagine.safetensors"


def test_build_image_workflow_uses_override_checkpoint(tmp_path) -> None:
    cfg = PipelineConfig(output_dir=tmp_path, sdxl_checkpoint="default.safetensors")
    pipeline = ScenePipeline(cfg)
    scene = Scene(
        id=1, description="x", image_prompt="a cat",
        video_prompt="x", duration_seconds=3.0,
    )
    wf = pipeline._build_image_workflow(
        scene, "cinematic", seed=1, sdxl_checkpoint="animagine.safetensors"
    )
    ckpt = wf[orch._find_node_by_title(wf, "Load SDXL checkpoint")]["inputs"]["ckpt_name"]
    assert ckpt == "animagine.safetensors"


def test_negative_prompt_override_is_appended(tmp_path) -> None:
    """Global negative override from the UI must append to the scene's own
    negatives without losing the scenario-generated ones."""
    cfg = PipelineConfig(
        output_dir=tmp_path,
        negative_prompt_override="extra fingers, watermark",
    )
    pipeline = ScenePipeline(cfg)
    scene = Scene(
        id=1,
        description="x",
        image_prompt="a cat",
        video_prompt="x",
        duration_seconds=3.0,
        negative_prompt="low quality, blurry",
    )
    merged = pipeline._compose_negative_prompt(scene)
    assert "low quality, blurry" in merged
    assert "extra fingers, watermark" in merged
    # Scene's own negatives must come first (they are scene-specific).
    assert merged.index("low quality") < merged.index("extra fingers")


def test_negative_override_survives_empty_scene_negative(tmp_path) -> None:
    cfg = PipelineConfig(output_dir=tmp_path, negative_prompt_override="text, hands")
    pipeline = ScenePipeline(cfg)
    scene = Scene(
        id=1, description="x", image_prompt="a cat",
        video_prompt="x", duration_seconds=3.0, negative_prompt="",
    )
    assert pipeline._compose_negative_prompt(scene) == "text, hands"


def test_apply_idea_hints_injects_scene_count_and_duration(tmp_path) -> None:
    cfg = PipelineConfig(output_dir=tmp_path, scenes_count_hint=6, scene_duration_hint=5.0)
    pipeline = ScenePipeline(cfg)
    out = pipeline._apply_idea_hints("a cat plays chess")
    assert "a cat plays chess" in out
    assert "6 scenes" in out
    assert "5 seconds" in out


def test_apply_idea_hints_is_noop_when_no_hints(tmp_path) -> None:
    cfg = PipelineConfig(output_dir=tmp_path)
    pipeline = ScenePipeline(cfg)
    assert pipeline._apply_idea_hints("hello") == "hello"


def test_fast_preview_caps_scenes_to_three_and_duration_to_two(tmp_path) -> None:
    cfg = PipelineConfig(
        output_dir=tmp_path, fast_preview=True, image_steps=28,
    )
    pipeline = ScenePipeline(cfg)
    scenes = [
        Scene(id=i, description="x", image_prompt="x", video_prompt="x", duration_seconds=6.0)
        for i in range(1, 7)
    ]
    scenario = Scenario(
        title="t", style="s", character_sheet="c", scenes=scenes,
    )
    trimmed = pipeline._apply_fast_preview(scenario)
    assert len(trimmed.scenes) == 3
    assert all(sc.duration_seconds <= 2.0 for sc in trimmed.scenes)
    # Docstring contract: fast_preview also halves SDXL steps.
    assert pipeline.config.image_steps == 14


def test_fast_preview_is_idempotent_across_repeated_calls(tmp_path) -> None:
    """Repeated run()/_apply_fast_preview() calls on the same pipeline
    must not keep halving image_steps (was 28 -> 14 -> 7 -> 3 -> 1)."""
    cfg = PipelineConfig(
        output_dir=tmp_path, fast_preview=True, image_steps=28,
    )
    pipeline = ScenePipeline(cfg)
    scenes = [
        Scene(id=i, description="x", image_prompt="x", video_prompt="x", duration_seconds=6.0)
        for i in range(1, 5)
    ]
    for _ in range(5):
        scenario = Scenario(title="t", style="s", character_sheet="c", scenes=list(scenes))
        pipeline._apply_fast_preview(scenario)
    assert pipeline.config.image_steps == 14


def test_fast_preview_disabled_leaves_scenario_intact(tmp_path) -> None:
    cfg = PipelineConfig(output_dir=tmp_path, fast_preview=False)
    pipeline = ScenePipeline(cfg)
    scenes = [
        Scene(id=i, description="x", image_prompt="x", video_prompt="x", duration_seconds=6.0)
        for i in range(1, 6)
    ]
    scenario = Scenario(title="t", style="s", character_sheet="c", scenes=scenes)
    out = pipeline._apply_fast_preview(scenario)
    assert len(out.scenes) == 5
    assert all(sc.duration_seconds == 6.0 for sc in out.scenes)
