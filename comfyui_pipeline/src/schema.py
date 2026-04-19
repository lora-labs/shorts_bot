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


class Scenario(BaseModel):
    """Top-level scenario returned by the Qwen stage."""

    title: str = Field(..., min_length=1)
    style: str = Field("cinematic", description="Global visual style keywords")
    scenes: list[Scene] = Field(..., min_length=1)

    @field_validator("scenes")
    @classmethod
    def _reindex(cls, scenes: list[Scene]) -> list[Scene]:
        for expected, scene in enumerate(scenes, start=1):
            if scene.id != expected:
                scene.id = expected
        return scenes
