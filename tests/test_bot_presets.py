"""Tests for the Telegram bot preset registry, config routing, and the
multi-step conversation wizard.

The bot itself is asyncio + python-telegram-bot which is awkward to
unit-test end-to-end, but everything that actually decides behaviour —
the preset registry, ``_build_config``, command→preset mapping, inline
idea extraction, and the wizard's keyboard factories — is pure Python
and worth locking down. This file is what guarantees a
``/generate_anime`` command actually causes the orchestrator to load
the anime SDXL checkpoint, and that the new 3-20 s duration picker
actually flows into ``PipelineConfig.total_duration_hint``.
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


# --------------------------------------------------------------------------- #
# Preset registry
# --------------------------------------------------------------------------- #


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


def test_build_config_propagates_total_duration_and_scenes_count(tmp_path) -> None:
    """The wizard collects total_duration + scenes_count and passes them
    through to PipelineConfig so Qwen actually receives the hint."""
    cfg = bot._build_config(
        tmp_path, bot.PRESETS["cinematic"],
        total_duration=12.0, scenes_count=4,
    )
    assert cfg.total_duration_hint == 12.0
    assert cfg.scenes_count_hint == 4


def test_build_config_leaves_hints_none_when_auto(tmp_path) -> None:
    cfg = bot._build_config(
        tmp_path, bot.PRESETS["default"],
        total_duration=None, scenes_count=None,
    )
    assert cfg.total_duration_hint is None
    assert cfg.scenes_count_hint is None


# --------------------------------------------------------------------------- #
# Command → preset routing
# --------------------------------------------------------------------------- #


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
        ("/generate_anime ninja cat in Tokyo", "anime"),  # inline idea keeps preset
    ],
)
def test_preset_key_from_command(command: str, expected_key: str) -> None:
    assert bot._preset_key_from_command(command) == expected_key


@pytest.mark.parametrize(
    "text,expected",
    [
        ("/generate_anime ninja cat", "ninja cat"),
        ("/generate_anime  spaced   idea  ", "spaced   idea"),
        ("/generate_anime", ""),
        ("/start", ""),
        ("just a plain message", "just a plain message"),
        ("", ""),
    ],
)
def test_inline_idea_extraction(text: str, expected: str) -> None:
    assert bot._inline_idea(text) == expected


# --------------------------------------------------------------------------- #
# Wizard keyboards
# --------------------------------------------------------------------------- #


def test_preset_keyboard_exposes_every_listed_preset() -> None:
    """The preset picker must show exactly the presets listed in
    ``PRESET_PICKER_ORDER`` — adding a new preset without wiring it into
    the picker order is a common mistake, this test catches it."""
    kb = bot._preset_keyboard()
    keys_on_buttons: list[str] = []
    for row in kb.inline_keyboard:
        for btn in row:
            assert btn.callback_data.startswith("preset:")
            keys_on_buttons.append(btn.callback_data.split(":", 1)[1])
    assert keys_on_buttons == list(bot.PRESET_PICKER_ORDER)
    for key in keys_on_buttons:
        assert key in bot.PRESETS


def test_duration_keyboard_covers_3_to_20_seconds_plus_auto() -> None:
    """User-request-driven regression: the duration picker must offer
    3-20 s and an 'Auto' escape hatch."""
    kb = bot._duration_keyboard()
    callback_values: list[str] = []
    for row in kb.inline_keyboard:
        for btn in row:
            assert btn.callback_data.startswith("dur:")
            callback_values.append(btn.callback_data.split(":", 1)[1])
    assert "3" in callback_values
    assert "20" in callback_values
    assert "auto" in callback_values
    # Every numeric option must parse as a float in [3, 20].
    numeric = [float(v) for v in callback_values if v != "auto"]
    assert min(numeric) >= 3.0
    assert max(numeric) <= 20.0


def test_scenes_keyboard_covers_expected_counts_plus_auto() -> None:
    kb = bot._scenes_keyboard()
    values: list[str] = []
    for row in kb.inline_keyboard:
        for btn in row:
            assert btn.callback_data.startswith("scenes:")
            values.append(btn.callback_data.split(":", 1)[1])
    assert {"3", "4", "5", "6", "auto"}.issubset(set(values))


def test_confirm_keyboard_offers_yes_no() -> None:
    kb = bot._confirm_keyboard()
    callbacks = [
        btn.callback_data
        for row in kb.inline_keyboard
        for btn in row
    ]
    assert "confirm:yes" in callbacks
    assert "confirm:no" in callbacks


# --------------------------------------------------------------------------- #
# Summary rendering
# --------------------------------------------------------------------------- #


def test_summary_renders_all_user_selections() -> None:
    data = {
        "preset": bot.PRESETS["cinematic"],
        "idea": "rainy Tokyo ninja",
        "total_duration": 10.0,
        "scenes_count": 4,
    }
    text = bot._summary(data)
    assert "Cinematic photo" in text
    assert "rainy Tokyo ninja" in text
    assert "10 s" in text
    assert "4" in text


def test_summary_renders_auto_when_hints_missing() -> None:
    data = {
        "preset": bot.PRESETS["default"],
        "idea": "anything",
        "total_duration": None,
        "scenes_count": None,
    }
    text = bot._summary(data)
    assert "auto" in text.lower()


@pytest.mark.parametrize(
    "raw_idea",
    [
        "my_cool_idea",
        "hello *world*",
        "look: `code`",
        "[link](http://example.com)",
        "under_score and *star* together",
    ],
)
def test_summary_escapes_markdown_in_user_idea(raw_idea: str) -> None:
    """``_summary`` is rendered with ``parse_mode='Markdown'`` so any
    Markdown-special character in the free-text idea must be escaped,
    otherwise Telegram rejects the message with BadRequest and the
    confirm step silently deadlocks."""
    from telegram.helpers import escape_markdown

    data = {
        "preset": bot.PRESETS["default"],
        "idea": raw_idea,
        "total_duration": 5.0,
        "scenes_count": 3,
    }
    text = bot._summary(data)
    expected = escape_markdown(raw_idea, version=1)
    assert expected in text
    # The italic wrapper the function chooses for the idea field uses a
    # single ``_`` on each side. Any un-escaped underscore inside the
    # idea would break that wrapper; assert the escaped form is present
    # and the raw form is NOT (unless the idea contains no specials).
    if raw_idea != expected:
        assert f"_{raw_idea}_" not in text


# --------------------------------------------------------------------------- #
# Conversation handler wiring
# --------------------------------------------------------------------------- #


def test_conversation_handler_has_all_wizard_states() -> None:
    """The ConversationHandler must declare handlers for every wizard
    state so users never hit a dead end."""
    conv = bot._build_conv_handler()
    assert bot.CHOOSE_PRESET in conv.states
    assert bot.AWAIT_IDEA in conv.states
    assert bot.CHOOSE_DURATION in conv.states
    assert bot.CHOOSE_SCENES in conv.states
    assert bot.CONFIRM in conv.states


# --------------------------------------------------------------------------- #
# Stale-keyboard regression tests
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_handle_scenes_choice_edits_message_to_clear_stale_buttons() -> None:
    """After the user picks a scene count, the scenes keyboard must be
    replaced with plain text; otherwise a late tap on the stale buttons
    fires an orphan callback in the CONFIRM state and flashes a client
    error. Mirrors the behaviour of ``_handle_duration_choice``."""
    from unittest.mock import AsyncMock, MagicMock

    query = MagicMock()
    query.data = "scenes:4"
    query.answer = AsyncMock()
    query.edit_message_text = AsyncMock()

    reply_msg = MagicMock()
    reply_msg.reply_text = AsyncMock()
    query.message = reply_msg

    update = MagicMock()
    update.callback_query = query
    update.effective_message = reply_msg
    update.message = None

    context = MagicMock()
    context.user_data = {
        "preset": bot.PRESETS["default"],
        "idea": "cat on mars",
        "total_duration": 10.0,
    }

    next_state = await bot._handle_scenes_choice(update, context)

    assert context.user_data["scenes_count"] == 4
    query.edit_message_text.assert_awaited_once()
    # The next state should be CONFIRM (via _ask_confirm, which sends a
    # fresh message with the confirm keyboard).
    assert next_state == bot.CONFIRM
    reply_msg.reply_text.assert_awaited()


@pytest.mark.asyncio
async def test_handle_scenes_choice_auto_edits_message() -> None:
    from unittest.mock import AsyncMock, MagicMock

    query = MagicMock()
    query.data = "scenes:auto"
    query.answer = AsyncMock()
    query.edit_message_text = AsyncMock()
    reply_msg = MagicMock()
    reply_msg.reply_text = AsyncMock()
    query.message = reply_msg

    update = MagicMock()
    update.callback_query = query
    update.effective_message = reply_msg
    update.message = None

    context = MagicMock()
    context.user_data = {
        "preset": bot.PRESETS["default"],
        "idea": "cat on mars",
        "total_duration": None,
    }

    await bot._handle_scenes_choice(update, context)

    assert context.user_data["scenes_count"] is None
    query.edit_message_text.assert_awaited_once()
    args, kwargs = query.edit_message_text.call_args
    rendered = args[0] if args else kwargs.get("text", "")
    assert "auto" in rendered.lower()
