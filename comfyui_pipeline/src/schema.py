"""Pydantic models for the scenario JSON produced by Qwen."""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class Scene(BaseModel):
    """A single scene in a generated scenario."""

    id: int = Field(..., ge=1, description="1-based scene index")
    description: str = Field(..., min_length=1, description="Short narrative description of the scene")
    image_prompt: str = Field(
        ...,
        min_length=1,
        description="Detailed prompt for the SDXL keyframe image (style, subject, camera, lighting)",
    )
    video_prompt: str = Field(
        ...,
        min_length=1,
        description="Motion description for LTX image-to-video (what moves, how, camera motion)",
    )
    duration_seconds: float = Field(4.0, gt=0, le=20, description="Target scene duration")
    negative_prompt: str = Field(
        "low quality, blurry, watermark, text, deformed",
        description="Negative prompt shared by both SDXL and LTX for this scene",
    )


# Canonical style preset ids. Qwen is asked to pick one of these.
# Orchestrator maps the id to a concrete SDXL checkpoint via
# ``PipelineConfig.checkpoint_presets`` (user-overridable).
STYLE_PRESETS = (
    "cinematic_photo",   # realistic photoreal (default fallback)
    "photoreal",         # alt photoreal (e.g. RealVisXL)
    "anime",             # anime/manga (AnimagineXL, Illustrious)
    "illustration",      # stylized illustration / 3D render / painterly
    "auto",              # let orchestrator pick based on prompt heuristics
)


class Scenario(BaseModel):
    """Top-level scenario returned by the Qwen stage."""

    title: str = Field(..., min_length=1)
    style: str = Field("cinematic", description="Global visual style keywords")
    style_preset: str = Field(
        "auto",
        description=(
            "Which SDXL checkpoint family best fits this story: "
            "'cinematic_photo', 'photoreal', 'anime', 'illustration', or "
            "'auto'. The orchestrator uses this to route the scenario to "
            "the right checkpoint file. 'auto' means the orchestrator "
            "falls back to its configured default."
        ),
    )
    character_sheet: str = Field(
        "",
        description=(
            "Canonical description of the main subject(s) that must remain "
            "visually consistent across every scene (species, colors, clothing, "
            "distinguishing marks, etc.). Prepended automatically to every "
            "scene's image_prompt."
        ),
    )
    scenes: list[Scene] = Field(..., min_length=1)

    @field_validator("style_preset")
    @classmethod
    def _normalize_preset(cls, v: str) -> str:
        """Be lenient: accept any casing / spaces / hyphens and normalise to
        the canonical id, or fall back to 'auto' if unrecognised. Better to
        degrade gracefully than to reject a whole scenario for a typo."""
        cleaned = (v or "").strip().lower().replace("-", "_").replace(" ", "_")
        if cleaned in STYLE_PRESETS:
            return cleaned
        return "auto"

    @field_validator("scenes")
    @classmethod
    def _reindex(cls, scenes: list[Scene]) -> list[Scene]:
        for expected, scene in enumerate(scenes, start=1):
            if scene.id != expected:
                scene.id = expected
        return scenes
