"""End-to-end orchestrator.

Flow
----
1. Load the Qwen script-gen workflow, patch the user idea and system prompt.
2. Submit to ComfyUI; collect the generated JSON string from history.
3. Parse the JSON into ``Scenario`` (Pydantic). Retry once with a stricter
   reminder if parsing fails.
4. For each scene:
   a. Patch ``scene_image_api.json`` with the scene's image prompt + seed.
   b. Run it; download the resulting image (also uploaded to ComfyUI /input
      so the next workflow can ``LoadImage`` it).
   c. Patch ``scene_video_api.json`` with the image filename and motion prompt.
   d. Run it; download the resulting mp4 into the output directory.
5. Write ``scenario.json`` and an ``index.json`` manifest listing each scene's
   artifacts.
"""
from __future__ import annotations

import copy
import json
import logging
import random
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests
from pydantic import ValidationError

from .comfy_client import ComfyClient, ComfyOutputFile
from .schema import Scenario, Scene

log = logging.getLogger(__name__)

WORKFLOW_DIR = Path(__file__).resolve().parent.parent / "workflows"
PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


@dataclass
class PipelineConfig:
    """Runtime configuration for the orchestrator."""

    comfyui_url: str = "http://127.0.0.1:8188"
    output_dir: Path = Path("output/scenes")
    sdxl_checkpoint: str = "juggernautXL_ragnarokBy.safetensors"
    # Style-preset routing table. Qwen tags each scenario with a preset
    # ('cinematic_photo', 'anime', etc.); the orchestrator looks the tag
    # up here to pick which SDXL checkpoint renders the scenes. Files
    # that are not present on disk are skipped gracefully (the
    # configured ``sdxl_checkpoint`` is used as fallback). Override this
    # dict from a downstream call-site (e.g. CLI, Gradio) to match your
    # local ``models/checkpoints/`` layout.
    checkpoint_presets: dict[str, str] = field(
        default_factory=lambda: {
            "cinematic_photo": "juggernautXL_ragnarokBy.safetensors",
            "photoreal": "realvisxlV50_v50Bakedvae.safetensors",
            "anime": "illustriousXL20_v20.safetensors",
            "illustration": "dreamshaperXL_alpha2Xl10.safetensors",
        }
    )
    # LoRA stacks applied on top of each preset's checkpoint. Each entry
    # is a list of ``(lora_filename, strength)`` tuples; strengths apply
    # to both MODEL and CLIP. Presets not listed here render without any
    # LoRAs. Missing files fall through with a warning (same graceful
    # degradation as ``checkpoint_presets``) so a partial model download
    # doesn't kill the whole run. Override from the call-site to match
    # your local ``models/loras/`` layout.
    preset_loras: dict[str, list[tuple[str, float]]] = field(
        default_factory=lambda: {
            # Cinematic look: film stock emulation + sharpness boost.
            "cinematic_photo": [
                ("SDXL_FILM_PHOTOGRAPHY_STYLE_V1.safetensors", 0.6),
                ("add-detail-xl.safetensors", 0.3),
            ],
            # General detail tweaker for photoreal / stylized renders.
            "photoreal": [("add-detail-xl.safetensors", 0.3)],
            "illustration": [("add-detail-xl.safetensors", 0.3)],
        }
    )
    # Directory on the ComfyUI machine that contains SDXL LoRAs. Used
    # for the same existence check as ``sdxl_checkpoints_dir``. ``None``
    # disables the check.
    sdxl_loras_dir: str | None = None
    # Optional user override: when set, wins over the scenario's own
    # ``style_preset``. Gradio / Telegram bot expose this as a dropdown
    # so the user can force a specific look. ``None`` or ``"auto"``
    # means "respect what Qwen picked".
    style_preset_override: str | None = None
    # Directory on the ComfyUI machine that contains SDXL checkpoints.
    # Used to verify a preset-selected file exists before switching to
    # it. ``None`` disables the check (the orchestrator trusts the user
    # and lets ComfyUI raise a clearer error if the file is missing).
    sdxl_checkpoints_dir: str | None = None
    ltx_checkpoint: str = "ltx-2.3-22b-distilled-fp8.safetensors"
    ltx_lora: str = "ltx-2.3-22b-distilled-lora-384.safetensors"
    ltx_lora_strength: float = 1.0
    ltx_text_encoder: str = "gemma_3_12B_it_fp4_mixed.safetensors"
    qwen_model: str = "Qwen/Qwen3-8B"
    # Where the Qwen LLM runs. On low-VRAM cards (e.g. RTX 4070 12 GB) the
    # 8B model in fp16 does not fit alongside SDXL/LTX; use "cpu" to generate
    # the scenario on CPU (~1-2 min) and keep the GPU free for diffusion,
    # or "auto" to let accelerate split layers across CPU+GPU.
    qwen_device: str = "auto"
    # Whether the Qwen node keeps the model resident in memory after
    # generating the scenario. Default False so VRAM/RAM is released for
    # the following SDXL + LTX stages.
    qwen_keep_loaded: bool = False

    # Vertical 9:16 (TikTok / Reels / Shorts). Both SDXL and LTX-2.3
    # accept these dimensions natively — 768x1344 is one of SDXL's
    # trained aspect buckets, and LTX needs dims divisible by 32
    # (544 = 32*17, 960 = 32*30) with a frames+1 length that's a
    # multiple of 8 (handled by _frames_for_duration).
    image_width: int = 768
    image_height: int = 1344
    video_width: int = 544
    video_height: int = 960
    video_length_frames: int = 121
    video_fps: float = 25.0

    image_steps: int = 28
    image_cfg: float = 6.5
    video_steps: int = 15
    video_cfg: float = 1.0

    # IP-Adapter (cross-scene character consistency). After scene 1 renders
    # we feed its SDXL output back into every subsequent scene through
    # IPAdapterAdvanced, which patches the SDXL MODEL with image-derived
    # tokens from CLIP-ViT-H. The character_sheet text prompt tells the
    # model WHAT the character is; IP-Adapter shows it what the character
    # already LOOKED like in scene 1. Together they lock the subject.
    use_ip_adapter: bool = True
    ip_adapter_model: str = "ip-adapter-plus_sdxl_vit-h.safetensors"
    clip_vision_model: str = "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"
    # 0.65 + "standard" + end_at 0.8 is the sweet spot for keeping the
    # same face/subject across scenes without copying the reference's
    # framing. History of this tuning:
    #   - 0.7 + "linear" (initial): scenes were near-duplicates of scene 1
    #   - 0.55 + "strong style transfer" (PR #15): style locked but
    #     identity drifted — faces of the protagonist changed every scene
    #   - 0.65 + "standard" + end_at 0.8 (now): "standard" weight_type
    #     preserves identity like the original "linear" but without the
    #     composition copy; stopping IPA at 80% of sampling lets fine
    #     detail (exact expression, pose, background) diverge per scene.
    # Tune up (↑0.8) for tighter identity (risk: scenes look similar),
    # down (↓0.5) for more per-scene freedom (risk: face drifts).
    ip_adapter_weight: float = 0.65
    ip_adapter_weight_type: str = "standard"
    ip_adapter_start_at: float = 0.0
    # Stop IPA at 80% so the last 20% of denoising is prompt-only —
    # avoids the "every scene has the same micro-expression" artifact
    # without losing identity (which is locked in the earlier steps).
    ip_adapter_end_at: float = 0.8

    # User-facing overrides applied at orchestration time.
    #
    # ``negative_prompt_override`` is appended (comma-joined) to every
    # scene's negative prompt. Useful to globally ban e.g. "text,
    # watermark, hands" without editing the system prompt.
    negative_prompt_override: str = ""
    # ``fast_preview`` halves SDXL steps, caps scene count to 3, and
    # clamps per-scene duration to 2s. Meant for rapid iteration before
    # committing to a full-length render.
    fast_preview: bool = False
    # Hints injected into the user idea before sending to Qwen. None
    # means "let Qwen decide" (legacy behaviour).
    scenes_count_hint: int | None = None
    scene_duration_hint: float | None = None

    # Ask ComfyUI to evict loaded models (via POST /free) at specific
    # stage boundaries. Needed on low-VRAM setups where SDXL+IP-Adapter
    # and LTX 22B fp8 cannot co-exist in VRAM (e.g. RTX 4070 12 GB):
    # without eviction the second stage starts swapping weights from
    # system RAM and the pipeline effectively hangs on the first video
    # step. Off by default would regress those users; leave on.
    free_vram_between_stages: bool = True

    # Scenario workflow can take 15+ min when Qwen runs on CPU (default on
    # low-VRAM setups). 30 min default gives headroom for that + reasoning.
    script_timeout: float = 1800.0
    scene_timeout: float = 1800.0

    seed: int | None = None
    max_script_retries: int = 1

    # Final step: concatenate per-scene mp4 files into one movie via ffmpeg.
    concat_final_video: bool = True
    ffmpeg_binary: str = "ffmpeg"


@dataclass
class SceneArtifacts:
    scene: Scene
    image_path: Path
    video_path: Path


@dataclass
class PipelineResult:
    scenario: Scenario
    output_dir: Path
    scene_artifacts: list[SceneArtifacts] = field(default_factory=list)
    final_video_path: Path | None = None


class PipelineError(RuntimeError):
    """Raised on unrecoverable orchestration failures."""


# --------------------------------------------------------------------------- #
# Workflow helpers
# --------------------------------------------------------------------------- #


def _load_workflow(name: str) -> dict[str, Any]:
    path = WORKFLOW_DIR / name
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _find_node_by_title(workflow: dict[str, Any], title: str) -> str:
    for node_id, node in workflow.items():
        if (node.get("_meta") or {}).get("title") == title:
            return node_id
    raise KeyError(f"No node with title {title!r} in workflow")


def _strip_json_fence(raw: str) -> str:
    """Best-effort clean-up of stray markdown fences from an LLM output."""
    stripped = raw.strip()
    fence_match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()
    return stripped


# --------------------------------------------------------------------------- #
# Main orchestrator
# --------------------------------------------------------------------------- #


class ScenePipeline:
    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self.client = ComfyClient(self.config.comfyui_url)
        self.config.output_dir = Path(self.config.output_dir)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        # Snapshot to keep fast_preview idempotent across repeated run() calls.
        self._original_image_steps = self.config.image_steps

    # ---------- Stage 1: scenario ----------

    def _build_script_workflow(self, idea: str, system_prompt: str, seed: int) -> dict[str, Any]:
        wf = copy.deepcopy(_load_workflow("script_gen_api.json"))
        sys_id = _find_node_by_title(wf, "System prompt")
        user_id = _find_node_by_title(wf, "User idea")
        qwen_id = _find_node_by_title(wf, "Qwen scenario generator")
        wf[sys_id]["inputs"]["value"] = system_prompt
        wf[user_id]["inputs"]["value"] = idea
        wf[qwen_id]["inputs"]["model_name_or_path"] = self.config.qwen_model
        wf[qwen_id]["inputs"]["device"] = self.config.qwen_device
        wf[qwen_id]["inputs"]["keep_loaded"] = self.config.qwen_keep_loaded
        wf[qwen_id]["inputs"]["seed"] = seed
        return wf

    def generate_scenario(self, idea: str) -> Scenario:
        system_prompt = (PROMPTS_DIR / "system_prompt.txt").read_text(encoding="utf-8")
        last_error: Exception | None = None
        attempts = 0
        current_idea = idea
        max_attempts = 1 + max(0, self.config.max_script_retries)
        while attempts < max_attempts:
            attempts += 1
            seed = self._seed() + attempts
            wf = self._build_script_workflow(current_idea, system_prompt, seed)
            log.info("Generating scenario (attempt %s/%s) seed=%s", attempts, max_attempts, seed)
            history = self.client.run(wf, timeout=self.config.script_timeout)

            qwen_node_id = _find_node_by_title(wf, "Qwen scenario generator")
            texts = self.client.collect_texts(history).get(qwen_node_id, [])
            if not texts:
                last_error = PipelineError(
                    f"Qwen node {qwen_node_id} produced no text output. History: {history.get('outputs')}"
                )
                continue
            raw = _strip_json_fence(texts[-1])
            try:
                payload = json.loads(raw)
                return Scenario.model_validate(payload)
            except (json.JSONDecodeError, ValidationError) as exc:
                log.warning("Scenario JSON invalid on attempt %s: %s", attempts, exc)
                last_error = exc
                current_idea = (
                    idea
                    + "\n\nIMPORTANT: your previous answer did not produce valid JSON matching "
                    "the schema. Return ONLY a JSON object. Error was: "
                    + str(exc)[:300]
                )

        raise PipelineError(f"Failed to get valid scenario JSON: {last_error}")

    # ---------- Stage 2: scene image ----------

    def _compose_image_prompt(
        self, scene: Scene, global_style: str, character_sheet: str
    ) -> str:
        """Prepend the canonical character description and global style to
        the scene's image prompt. Keeping appearance details in ONE place
        (character_sheet) and reusing them verbatim across all scenes is
        what gives us the same cat / person / robot in every frame.
        """
        parts: list[str] = []
        if character_sheet and character_sheet.strip():
            parts.append(character_sheet.strip())
        parts.append(scene.image_prompt.strip())
        if global_style and global_style.strip():
            parts.append(global_style.strip())
        return ", ".join(parts)

    def _compose_negative_prompt(self, scene: Scene) -> str:
        """Merge the scene's scenario-generated negatives with the user's
        global override (``PipelineConfig.negative_prompt_override``).
        Both are joined with ``, ``; empty sides are skipped.
        """
        parts: list[str] = []
        if scene.negative_prompt and scene.negative_prompt.strip():
            parts.append(scene.negative_prompt.strip())
        override = (self.config.negative_prompt_override or "").strip()
        if override:
            parts.append(override)
        return ", ".join(parts)

    def _compose_video_prompt(
        self, scene: Scene, global_style: str, character_sheet: str
    ) -> str:
        """LTX also benefits from seeing the character description so it
        does not drift away from the keyframe's subject during motion.
        """
        parts: list[str] = []
        if character_sheet and character_sheet.strip():
            parts.append(character_sheet.strip())
        parts.append(scene.video_prompt.strip())
        if global_style and global_style.strip():
            parts.append(global_style.strip())
        return ", ".join(parts)

    def _resolve_preset_name(self, scenario_preset: str | None) -> str:
        """Resolve the effective preset name (post-override, post-normalisation).

        Returns either a concrete preset id ('cinematic_photo', 'anime', …)
        or ``'auto'`` when the caller should fall back to the configured
        default checkpoint / no LoRAs.
        """
        # Mirror Scenario._normalize_preset so callers can pass sloppy
        # casing / spaces / hyphens from UI labels without silently
        # missing the checkpoint_presets key.
        override = (
            (self.config.style_preset_override or "")
            .strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )
        preset = override if override and override != "auto" else (scenario_preset or "auto")
        return preset or "auto"

    def _resolve_sdxl_checkpoint(self, scenario_preset: str | None) -> str:
        """Decide which SDXL checkpoint file to load for this run.

        Priority:
          1. ``config.style_preset_override`` (user-forced from UI / bot)
          2. Scenario's own ``style_preset`` (emitted by Qwen)
          3. ``config.sdxl_checkpoint`` fallback

        If the resolved preset's file is declared missing by the optional
        ``config.sdxl_checkpoints_dir`` existence check, we fall back to
        ``config.sdxl_checkpoint`` and log a warning so the user can tell
        routing was bypassed.
        """
        preset = self._resolve_preset_name(scenario_preset)
        if preset == "auto":
            return self.config.sdxl_checkpoint
        filename = self.config.checkpoint_presets.get(preset)
        if not filename:
            log.warning(
                "Style preset %r is not mapped in checkpoint_presets; "
                "using configured default %r.", preset, self.config.sdxl_checkpoint,
            )
            return self.config.sdxl_checkpoint
        if self.config.sdxl_checkpoints_dir:
            candidate = Path(self.config.sdxl_checkpoints_dir) / filename
            if not candidate.exists():
                log.warning(
                    "Preset %r -> %r not found on disk (checked %s); falling back to %r.",
                    preset, filename, candidate, self.config.sdxl_checkpoint,
                )
                return self.config.sdxl_checkpoint
        log.info("Style preset %r -> SDXL checkpoint %r", preset, filename)
        return filename

    def _resolve_preset_loras(
        self, scenario_preset: str | None
    ) -> list[tuple[str, float]]:
        """Return the LoRA stack to chain on top of the preset's checkpoint.

        Follows the same preset-resolution rules as ``_resolve_sdxl_checkpoint``.
        Entries whose file is missing (per optional ``sdxl_loras_dir``) are
        filtered out with a warning so a partial download doesn't abort the
        whole run.
        """
        preset = self._resolve_preset_name(scenario_preset)
        if preset == "auto":
            return []
        stack = self.config.preset_loras.get(preset) or []
        if not stack:
            return []
        verified: list[tuple[str, float]] = []
        for entry in stack:
            filename, strength = entry
            if self.config.sdxl_loras_dir:
                candidate = Path(self.config.sdxl_loras_dir) / filename
                if not candidate.exists():
                    log.warning(
                        "Preset %r LoRA %r not found on disk (checked %s); skipping.",
                        preset, filename, candidate,
                    )
                    continue
            verified.append((filename, float(strength)))
        if verified:
            log.info(
                "Style preset %r -> LoRA stack: %s",
                preset,
                ", ".join(f"{name}@{s:.2f}" for name, s in verified),
            )
        return verified

    @staticmethod
    def _inject_loras(
        wf: dict[str, Any],
        ckpt_id: str,
        loras: list[tuple[str, float]],
        consumers: list[tuple[str, str, int]],
    ) -> None:
        """Insert a chain of ``LoraLoader`` nodes between the checkpoint
        and the given consumers.

        ``consumers`` is a list of ``(node_id, input_name, slot)`` tuples
        that currently reference ``[ckpt_id, slot]`` and should be
        rewritten to read from the last LoRA in the chain. ``slot`` 0
        means MODEL, 1 means CLIP (matching SDXL CheckpointLoaderSimple
        outputs).
        """
        if not loras:
            return
        existing_ids = [int(k) for k in wf.keys() if str(k).isdigit()]
        next_id = max(existing_ids) + 1 if existing_ids else 100
        prev_model: list[Any] = [ckpt_id, 0]
        prev_clip: list[Any] = [ckpt_id, 1]
        for idx, (lora_name, strength) in enumerate(loras, start=1):
            node_id = str(next_id)
            next_id += 1
            wf[node_id] = {
                "class_type": "LoraLoader",
                "inputs": {
                    "lora_name": lora_name,
                    "strength_model": strength,
                    "strength_clip": strength,
                    "model": prev_model,
                    "clip": prev_clip,
                },
                "_meta": {"title": f"LoRA {idx}: {lora_name}"},
            }
            prev_model = [node_id, 0]
            prev_clip = [node_id, 1]
        for node_id, input_name, slot in consumers:
            wf[node_id]["inputs"][input_name] = (
                prev_model if slot == 0 else prev_clip
            )

    def _build_image_workflow(
        self,
        scene: Scene,
        global_style: str,
        seed: int,
        character_sheet: str = "",
        reference_image: str | None = None,
        sdxl_checkpoint: str | None = None,
        loras: list[tuple[str, float]] | None = None,
    ) -> dict[str, Any]:
        """Build the SDXL image workflow.

        If ``reference_image`` is a non-empty string (a filename already
        uploaded to ComfyUI's /input), load the IP-Adapter variant so the
        render is conditioned on that image in addition to the text prompt.
        Otherwise load the plain workflow (used for scene 1, which has no
        reference yet).

        ``loras`` is an optional list of ``(filename, strength)`` tuples
        applied on top of the checkpoint. The LoRAs are chained (first
        entry wraps the checkpoint, later entries wrap the previous
        LoRA's output) and their MODEL/CLIP outputs replace the direct
        checkpoint references in the downstream IPA / sampler / text
        encode nodes.
        """
        use_ipa = bool(reference_image) and self.config.use_ip_adapter
        wf_name = "scene_image_ipa_api.json" if use_ipa else "scene_image_api.json"
        wf = copy.deepcopy(_load_workflow(wf_name))
        ckpt_id = _find_node_by_title(wf, "Load SDXL checkpoint")
        latent_id = _find_node_by_title(wf, "Empty latent 9:16")
        pos_id = _find_node_by_title(wf, "Positive prompt")
        neg_id = _find_node_by_title(wf, "Negative prompt")
        sampler_id = _find_node_by_title(wf, "Sample")
        save_id = _find_node_by_title(wf, "Save scene image")

        wf[ckpt_id]["inputs"]["ckpt_name"] = sdxl_checkpoint or self.config.sdxl_checkpoint
        wf[latent_id]["inputs"]["width"] = self.config.image_width
        wf[latent_id]["inputs"]["height"] = self.config.image_height

        if use_ipa:
            ipa_model_id = _find_node_by_title(wf, "Load IP-Adapter")
            clip_vision_id = _find_node_by_title(wf, "Load CLIP Vision")
            load_ref_id = _find_node_by_title(wf, "Load reference image")
            apply_ipa_id = _find_node_by_title(wf, "Apply IP-Adapter")
            wf[ipa_model_id]["inputs"]["ipadapter_file"] = self.config.ip_adapter_model
            wf[clip_vision_id]["inputs"]["clip_name"] = self.config.clip_vision_model
            wf[load_ref_id]["inputs"]["image"] = reference_image
            wf[apply_ipa_id]["inputs"]["weight"] = self.config.ip_adapter_weight
            wf[apply_ipa_id]["inputs"]["weight_type"] = self.config.ip_adapter_weight_type
            wf[apply_ipa_id]["inputs"]["start_at"] = self.config.ip_adapter_start_at
            wf[apply_ipa_id]["inputs"]["end_at"] = self.config.ip_adapter_end_at

        if loras:
            # IPA node wraps MODEL, sampler reads it; text encoders read
            # CLIP directly from the checkpoint. Rewire only the nodes
            # that currently point at the checkpoint outputs.
            consumers: list[tuple[str, str, int]] = [
                (pos_id, "clip", 1),
                (neg_id, "clip", 1),
            ]
            if use_ipa:
                consumers.append((apply_ipa_id, "model", 0))
            else:
                consumers.append((sampler_id, "model", 0))
            self._inject_loras(wf, ckpt_id, loras, consumers)

        wf[pos_id]["inputs"]["text"] = self._compose_image_prompt(
            scene, global_style, character_sheet
        )
        wf[neg_id]["inputs"]["text"] = self._compose_negative_prompt(scene)

        wf[sampler_id]["inputs"]["noise_seed"] = seed
        wf[sampler_id]["inputs"]["steps"] = self.config.image_steps
        wf[sampler_id]["inputs"]["end_at_step"] = max(self.config.image_steps, 10000)
        wf[sampler_id]["inputs"]["cfg"] = self.config.image_cfg
        wf[save_id]["inputs"]["filename_prefix"] = f"scene_{scene.id:02d}_image"
        return wf

    def _render_scene_image(
        self,
        scene: Scene,
        global_style: str,
        seed: int,
        character_sheet: str = "",
        reference_image: str | None = None,
        sdxl_checkpoint: str | None = None,
        loras: list[tuple[str, float]] | None = None,
    ) -> tuple[ComfyOutputFile, Path]:
        wf = self._build_image_workflow(
            scene, global_style, seed, character_sheet, reference_image,
            sdxl_checkpoint=sdxl_checkpoint,
            loras=loras,
        )
        save_id = _find_node_by_title(wf, "Save scene image")
        history = self.client.run(wf, timeout=self.config.scene_timeout)

        outputs = self.client.collect_outputs(history)
        files = outputs.get(save_id) or []
        if not files:
            raise PipelineError(f"No image output from SaveImage node for scene {scene.id}")
        image = files[0]
        local_path = self.config.output_dir / f"scene_{scene.id:02d}.png"
        self.client.fetch_file(image, local_path)
        return image, local_path

    # ---------- Stage 3: scene video ----------

    def _upload_image_as_input(self, local_path: Path) -> str:
        """Upload an image to ComfyUI /input via /upload/image and return its filename."""
        with local_path.open("rb") as fh:
            files = {"image": (local_path.name, fh, "image/png")}
            data = {"overwrite": "true"}
            resp = requests.post(
                f"{self.client.base_url}/upload/image",
                files=files,
                data=data,
                timeout=120,
            )
        resp.raise_for_status()
        payload = resp.json()
        name = payload.get("name") or local_path.name
        subfolder = payload.get("subfolder") or ""
        return f"{subfolder}/{name}" if subfolder else name

    def _build_video_workflow(
        self,
        scene: Scene,
        input_image_name: str,
        global_style: str,
        seed: int,
        character_sheet: str = "",
    ) -> dict[str, Any]:
        wf = copy.deepcopy(_load_workflow("scene_video_api.json"))
        ckpt_id = _find_node_by_title(wf, "Load LTX checkpoint")
        enc_id = _find_node_by_title(wf, "LTX text encoder")
        lora_id = _find_node_by_title(wf, "Apply LTX distilled LoRA")
        image_id = _find_node_by_title(wf, "Load keyframe image")
        pos_id = _find_node_by_title(wf, "Video positive prompt")
        neg_id = _find_node_by_title(wf, "Video negative prompt")
        cond_id = _find_node_by_title(wf, "LTX conditioning")
        # LTX 2.3 is an audio-visual model. The sampler expects a unified
        # AV-latent (video latent + audio latent concatenated). The workflow
        # therefore fans out into two parallel branches:
        #   • video: EmptyLTXVLatentVideo → LTXVImgToVideoConditionOnly
        #   • audio: LTXVAudioVAELoader → LTXVEmptyLatentAudio
        # Both branches merge in LTXVConcatAVLatent before sampling. After
        # sampling, LTXVSeparateAVLatent splits them again; we decode only
        # the video latent and discard the (silent) audio latent.
        video_latent_id = _find_node_by_title(wf, "Empty video latent")
        audio_vae_id = _find_node_by_title(wf, "Load audio VAE")
        audio_latent_id = _find_node_by_title(wf, "Empty audio latent")
        sched_id = _find_node_by_title(wf, "LTX scheduler")
        noise_id = _find_node_by_title(wf, "Random noise")
        guider_id = _find_node_by_title(wf, "CFG guider")
        create_id = _find_node_by_title(wf, "Create video from frames")
        save_id = _find_node_by_title(wf, "Save scene video")

        wf[ckpt_id]["inputs"]["ckpt_name"] = self.config.ltx_checkpoint
        # LTXAVTextEncoderLoader needs BOTH the Gemma encoder file and the
        # LTX checkpoint (to read the cross-attention projection weights that
        # pair Gemma embeddings with the video model).
        wf[enc_id]["inputs"]["text_encoder"] = self.config.ltx_text_encoder
        wf[enc_id]["inputs"]["ckpt_name"] = self.config.ltx_checkpoint
        # The audio VAE loader re-reads the same LTX checkpoint — the audio
        # VAE weights live in the same .safetensors alongside the video model.
        wf[audio_vae_id]["inputs"]["ckpt_name"] = self.config.ltx_checkpoint
        wf[lora_id]["inputs"]["lora_name"] = self.config.ltx_lora
        wf[lora_id]["inputs"]["strength_model"] = self.config.ltx_lora_strength
        wf[image_id]["inputs"]["image"] = input_image_name

        wf[pos_id]["inputs"]["text"] = self._compose_video_prompt(
            scene, global_style, character_sheet
        )
        wf[neg_id]["inputs"]["text"] = self._compose_negative_prompt(scene)

        length = self._frames_for_duration(scene.duration_seconds)
        wf[cond_id]["inputs"]["frame_rate"] = self.config.video_fps
        wf[video_latent_id]["inputs"]["width"] = self.config.video_width
        wf[video_latent_id]["inputs"]["height"] = self.config.video_height
        wf[video_latent_id]["inputs"]["length"] = length
        wf[audio_latent_id]["inputs"]["frames_number"] = length
        wf[audio_latent_id]["inputs"]["frame_rate"] = int(round(self.config.video_fps))
        wf[sched_id]["inputs"]["steps"] = self.config.video_steps
        wf[noise_id]["inputs"]["noise_seed"] = seed
        wf[guider_id]["inputs"]["cfg"] = self.config.video_cfg
        wf[create_id]["inputs"]["fps"] = self.config.video_fps
        wf[save_id]["inputs"]["filename_prefix"] = f"scene_{scene.id:02d}_video"
        return wf

    def _frames_for_duration(self, seconds: float) -> int:
        # LTX requires (length - 1) % 8 == 0 and length >= 9.
        target = max(9, int(round(seconds * self.config.video_fps)))
        remainder = (target - 1) % 8
        if remainder:
            target += 8 - remainder
        return target

    def _render_scene_video(
        self,
        scene: Scene,
        input_image_name: str,
        global_style: str,
        seed: int,
        character_sheet: str = "",
    ) -> Path:
        wf = self._build_video_workflow(
            scene, input_image_name, global_style, seed, character_sheet
        )
        save_id = _find_node_by_title(wf, "Save scene video")
        history = self.client.run(wf, timeout=self.config.scene_timeout)
        outputs = self.client.collect_outputs(history)
        files = outputs.get(save_id) or []
        if not files:
            raise PipelineError(f"No video output from SaveVideo node for scene {scene.id}")
        video = files[0]
        local_path = self.config.output_dir / f"scene_{scene.id:02d}.mp4"
        self.client.fetch_file(video, local_path)
        return local_path

    # ---------- Top-level ----------

    def _seed(self) -> int:
        return self.config.seed if self.config.seed is not None else random.randint(1, 2**31 - 1)

    def _apply_idea_hints(self, idea: str) -> str:
        """Append user-facing hints (scene count, duration) as plain-English
        guidance to the idea before sending it to Qwen.

        We prefer prompt-level hints over mutating the system prompt so the
        default behaviour (Qwen picks what fits the story) is preserved when
        the user does not opt in from the UI.
        """
        hints: list[str] = []
        n = self.config.scenes_count_hint
        if n is not None and n > 0:
            hints.append(f"Generate exactly {n} scenes.")
        d = self.config.scene_duration_hint
        if d is not None and d > 0:
            hints.append(f"Each scene should be about {d:g} seconds long.")
        if not hints:
            return idea
        return idea.strip() + "\n\nUser preferences:\n- " + "\n- ".join(hints)

    def _apply_fast_preview(self, scenario: Scenario) -> Scenario:
        """In fast-preview mode cap scenes to 3, duration to 2s, and halve
        SDXL steps so a full run completes in a couple of minutes instead
        of 30.
        """
        if not self.config.fast_preview:
            return scenario
        # Halve against the ORIGINAL step count (captured in __init__) so
        # calling run() repeatedly on the same ScenePipeline with
        # fast_preview=True doesn't compound (28 -> 14 -> 7 -> 3 -> 1).
        self.config.image_steps = max(1, self._original_image_steps // 2)
        trimmed = scenario.scenes[:3]
        for sc in trimmed:
            sc.duration_seconds = min(sc.duration_seconds, 2.0)
        scenario.scenes = trimmed
        log.info(
            "fast_preview: trimmed to %d scenes, <=2s each, image_steps=%d",
            len(trimmed), self.config.image_steps,
        )
        return scenario

    def run(self, idea: str) -> PipelineResult:
        log.info("=== ScenePipeline start: %r ===", idea)
        scenario = self.generate_scenario(self._apply_idea_hints(idea))
        scenario = self._apply_fast_preview(scenario)
        log.info("Scenario: %s (%d scenes)", scenario.title, len(scenario.scenes))

        # After Qwen has produced the scenario, ask ComfyUI to drop the
        # 8B LLM out of memory before we start loading SDXL checkpoints.
        # On combined CPU+GPU setups qwen_keep_loaded=False already does
        # this inside the custom node, but /free is a belt-and-braces
        # cleanup for any residual CUDA allocator state.
        if self.config.free_vram_between_stages:
            self.client.free_memory()

        (self.config.output_dir / "scenario.json").write_text(
            scenario.model_dump_json(indent=2), encoding="utf-8"
        )

        result = PipelineResult(scenario=scenario, output_dir=self.config.output_dir)
        base_seed = self._seed()

        # After scene 1 is rendered we feed its (already-uploaded) keyframe
        # to IPAdapterAdvanced for every subsequent scene. This locks the
        # visual identity of the main subject across the narrative without
        # requiring the user to supply a reference image.
        ip_adapter_reference: str | None = None
        # Resolve the SDXL checkpoint once per scenario (routing applies
        # uniformly to all scenes — switching mid-run would break identity
        # lock because IP-Adapter embeds are tuned to a specific checkpoint).
        active_checkpoint = self._resolve_sdxl_checkpoint(
            getattr(scenario, "style_preset", None)
        )
        active_loras = self._resolve_preset_loras(
            getattr(scenario, "style_preset", None)
        )

        for scene in scenario.scenes:
            log.info("[scene %d/%d] %s", scene.id, len(scenario.scenes), scene.description)
            img_seed = base_seed + 10 * scene.id
            vid_seed = base_seed + 10 * scene.id + 1

            _, image_local = self._render_scene_image(
                scene,
                scenario.style,
                img_seed,
                scenario.character_sheet,
                reference_image=ip_adapter_reference,
                sdxl_checkpoint=active_checkpoint,
                loras=active_loras,
            )
            input_name = self._upload_image_as_input(image_local)
            if self.config.use_ip_adapter and ip_adapter_reference is None:
                # The image uploaded as LTX keyframe is the same file we want
                # IPAdapter to see in later scenes — reuse it.
                ip_adapter_reference = input_name
            # Evict SDXL + IP-Adapter + LoRA weights before loading the
            # much larger LTX video model. On 12 GB cards (RTX 4070) the
            # two stages combined exceed VRAM and ComfyUI otherwise
            # silently stalls streaming weights from system RAM.
            if self.config.free_vram_between_stages:
                self.client.free_memory()
            video_local = self._render_scene_video(
                scene, input_name, scenario.style, vid_seed, scenario.character_sheet
            )
            # And free the LTX weights before looping back to the next
            # scene's SDXL pass, so the SDXL checkpoint + IPA load into
            # a clean VRAM rather than evicting pieces of LTX first.
            if self.config.free_vram_between_stages:
                self.client.free_memory()
            result.scene_artifacts.append(
                SceneArtifacts(scene=scene, image_path=image_local, video_path=video_local)
            )

        manifest = {
            "title": scenario.title,
            "style": scenario.style,
            "scenes": [
                {
                    "id": sa.scene.id,
                    "description": sa.scene.description,
                    "image": sa.image_path.name,
                    "video": sa.video_path.name,
                }
                for sa in result.scene_artifacts
            ],
        }
        (self.config.output_dir / "index.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        if self.config.concat_final_video and result.scene_artifacts:
            try:
                result.final_video_path = self._concat_final_video(result)
                log.info("Final video concatenated: %s", result.final_video_path)
            except Exception as exc:  # noqa: BLE001 - concat is best-effort
                log.warning("Final video concatenation failed: %s", exc)

        log.info("=== ScenePipeline done → %s ===", self.config.output_dir)
        return result

    # ---------- Stage 4: ffmpeg concat ----------

    def _concat_final_video(self, result: PipelineResult) -> Path:
        if not shutil.which(self.config.ffmpeg_binary):
            raise PipelineError(
                f"ffmpeg binary {self.config.ffmpeg_binary!r} not found in PATH"
            )
        if not result.scene_artifacts:
            raise PipelineError("no scenes to concatenate")

        list_path = self.config.output_dir / "concat_list.txt"
        with list_path.open("w", encoding="utf-8") as fh:
            for sa in result.scene_artifacts:
                escaped = str(sa.video_path.resolve()).replace("'", "'\\''")
                fh.write(f"file '{escaped}'\n")
        final_path = self.config.output_dir / "final.mp4"
        cmd = [
            self.config.ffmpeg_binary, "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(list_path),
            "-c", "copy",
            str(final_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            log.warning(
                "ffmpeg stream-copy concat failed (rc=%s), re-encoding. stderr tail: %s",
                proc.returncode, proc.stderr[-500:],
            )
            cmd_enc = [
                self.config.ffmpeg_binary, "-y",
                "-f", "concat", "-safe", "0",
                "-i", str(list_path),
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-preset", "medium", "-crf", "18",
                str(final_path),
            ]
            proc = subprocess.run(cmd_enc, capture_output=True, text=True)
            if proc.returncode != 0:
                raise PipelineError(
                    f"ffmpeg concat failed (rc={proc.returncode}): {proc.stderr[-500:]}"
                )
        return final_path


__all__ = ["PipelineConfig", "PipelineResult", "SceneArtifacts", "ScenePipeline", "PipelineError"]
