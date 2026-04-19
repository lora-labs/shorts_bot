"""Unit tests for the scenario Pydantic schema."""
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from comfyui_pipeline.src.schema import Scenario, Scene


def _valid_payload() -> dict:
    return {
        "title": "Robot paints",
        "style": "cinematic, warm palette",
        "scenes": [
            {
                "id": 1,
                "description": "Robot discovers a canvas.",
                "image_prompt": "a small robot in a sunlit studio, 35mm",
                "video_prompt": "robot slowly approaches canvas, dolly in",
                "duration_seconds": 4,
            },
            {
                "id": 2,
                "description": "Robot paints its first stroke.",
                "image_prompt": "close-up of robotic hand holding brush",
                "video_prompt": "hand moves brush across canvas, soft motion",
                "duration_seconds": 3.5,
            },
        ],
    }


def test_scenario_parses_valid_payload() -> None:
    scenario = Scenario.model_validate(_valid_payload())
    assert scenario.title == "Robot paints"
    assert len(scenario.scenes) == 2
    assert scenario.scenes[0].negative_prompt  # has a default


def test_scenario_reindexes_scene_ids() -> None:
    payload = _valid_payload()
    payload["scenes"][0]["id"] = 5
    payload["scenes"][1]["id"] = 9
    scenario = Scenario.model_validate(payload)
    assert [s.id for s in scenario.scenes] == [1, 2]


def test_scenario_rejects_empty_scenes() -> None:
    payload = _valid_payload()
    payload["scenes"] = []
    with pytest.raises(ValidationError):
        Scenario.model_validate(payload)


def test_scene_requires_non_empty_prompts() -> None:
    with pytest.raises(ValidationError):
        Scene.model_validate(
            {
                "id": 1,
                "description": "x",
                "image_prompt": "",
                "video_prompt": "motion",
                "duration_seconds": 3,
            }
        )


def test_scenario_roundtrip_json() -> None:
    scenario = Scenario.model_validate(_valid_payload())
    blob = scenario.model_dump_json()
    restored = Scenario.model_validate(json.loads(blob))
    assert restored == scenario
