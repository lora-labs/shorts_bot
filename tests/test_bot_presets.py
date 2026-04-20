"""Tests for the Telegram bot preset registry & config routing.

The bot itself is asyncio + python-telegram-bot which is awkward to unit-test
end-to-end, but the preset registry and ``_build_config`` are pure-Python
helpers worth locking down: they are what guarantees a ``/generate_anime``
command actually causes the orchestrator to load the anime SDXL checkpoint.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("telegram")

import bot  # noqa: E402


def test_preset_registry_has_all_expected_keys() -> None:
    expected = {"default", "anime", "cinematic", "photo", "illustration", "fast"}
    assert expected.issubset(set(bot.PRESETS.keys()))


def test_anime_preset_maps_to_anime_checkpoint(tmp_path) -> None:
    cfg = bot._build_config(tmp_path, bot.PRESETS["anime"])
    assert cfg.style_preset_override == "anime"
    assert cfg.fast_preview is False


def test_cinematic_preset_maps_to_cinematic_checkpoint(tmp_path) -> None:
    cfg = bot._build_config(tmp_path, bot.PRESETS["cinematic"])
    assert cfg.style_preset_override == "cinematic_photo"


def test_photo_preset_maps_to_photoreal(tmp_path) -> None:
    cfg = bot._build_config(tmp_path, bot.PRESETS["photo"])
    assert cfg.style_preset_override == "photoreal"


def test_illustration_preset_maps_to_illustration(tmp_path) -> None:
    cfg = bot._build_config(tmp_path, bot.PRESETS["illustration"])
    assert cfg.style_preset_override == "illustration"


def test_fast_preset_enables_fast_preview(tmp_path) -> None:
    cfg = bot._build_config(tmp_path, bot.PRESETS["fast"])
    assert cfg.fast_preview is True
    # Fast preview stacks on top of auto-routing, not a specific checkpoint.
    assert cfg.style_preset_override is None


def test_default_preset_is_inert(tmp_path) -> None:
    cfg = bot._build_config(tmp_path, bot.PRESETS["default"])
    assert cfg.style_preset_override is None
    assert cfg.fast_preview is False


def test_build_config_skips_unknown_fields_gracefully(tmp_path) -> None:
    """If a future preset references a PipelineConfig field that does not
    exist yet, ``_build_config`` must not crash — it should log a warning
    and keep the known fields. Guards against deploying the bot against
    a partially-migrated orchestrator."""
    rogue = bot.Preset(
        label="rogue",
        style_preset_override="anime",
        overrides={"nonexistent_field_xyz": 42},
    )
    cfg = bot._build_config(tmp_path, rogue)
    assert cfg.style_preset_override == "anime"
    assert not hasattr(cfg, "nonexistent_field_xyz")


class _FakeMessage:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeUpdate:
    def __init__(self, text: str) -> None:
        self.message = _FakeMessage(text)


@pytest.mark.parametrize(
    "command,expected_key",
    [
        ("/start", "default"),
        ("/generate", "default"),
        ("/generate_anime", "anime"),
        ("/generate_cinematic", "cinematic"),
        ("/generate_photo", "photo"),
        ("/generate_illustration", "illustration"),
        ("/generate_fast", "fast"),
        ("/generate_unknown", "default"),  # unknown suffix -> default, never crash
        ("/generate_anime@mybotname", "anime"),  # group-chat @-suffix is stripped
    ],
)
def test_preset_from_update_routes_command_to_preset(command: str, expected_key: str) -> None:
    preset = bot._preset_from_update(_FakeUpdate(command))
    assert preset is bot.PRESETS[expected_key]


def test_preset_from_update_with_inline_idea_still_picks_preset() -> None:
    """``/generate_anime ninja cat`` must still resolve to the anime preset,
    not fall back to default because of the trailing text."""
    preset = bot._preset_from_update(_FakeUpdate("/generate_anime ninja cat in Tokyo"))
    assert preset is bot.PRESETS["anime"]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("/generate_anime ninja cat", "ninja cat"),
        ("/generate_anime  spaced   idea  ", "spaced   idea"),
        ("/generate_anime", ""),
        ("/start", ""),
        ("just a plain message", "just a plain message"),
    ],
)
def test_inline_idea_extraction(text: str, expected: str) -> None:
    assert bot._inline_idea(_FakeUpdate(text)) == expected
