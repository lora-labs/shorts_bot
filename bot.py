"""Telegram bot entry point.

Conversation wizard
-------------------
The user is walked through a short multi-step dialog: **preset → idea →
duration → scenes → confirm**. Each step (except the free-text *idea*)
is an inline keyboard so the user just taps instead of typing.

Commands
--------
* ``/generate`` — full wizard (asks preset, then idea, duration, scenes).
* ``/generate_anime`` / ``/generate_cinematic`` / ``/generate_photo`` /
  ``/generate_illustration`` — skip the preset step, go straight to
  "опиши идею" with the preset already locked in.
* ``/generate_fast`` — fast-preview preset; skips the duration and
  scene-count pickers (both are overridden by fast-preview defaults).
* ``/start`` — alias for ``/generate`` (legacy).
* ``/cancel`` — abort the dialog at any point.
* ``/help`` — list of commands.

Each command accepts the idea either inline (``/generate_anime a ninja
cat in Tokyo``) or in a follow-up message. In the inline case the
``AWAIT_IDEA`` state is skipped.

Required environment
--------------------
* ``TELEGRAM_BOT_TOKEN`` — bot token from @BotFather.
* ``COMFYUI_URL`` (optional, default ``http://127.0.0.1:8188``).
* ``GOOGLE_APPLICATION_CREDENTIALS`` / ``credentials.json`` — only needed
  for the Google Drive fallback; the bot runs fine without it.
"""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)

from comfyui_pipeline.src.orchestrator import (
    PipelineConfig,
    PipelineError,
    ScenePipeline,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("shorts_bot")

API_KEY = os.environ.get("TELEGRAM_BOT_TOKEN", "")
COMFYUI_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188")
OUTPUT_DIR = Path(os.environ.get("BOT_OUTPUT_DIR", "output/bot_videos"))
CREDENTIALS_FILE = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "credentials.json")
DRIVE_FOLDER_ID = os.environ.get("DRIVE_FOLDER_ID") or None
TELEGRAM_MAX_UPLOAD_MB = 50  # Telegram Bot API hard limit for outgoing files.

# Conversation states. Each represents the question the bot is currently
# waiting for an answer to.
CHOOSE_PRESET = 0
AWAIT_IDEA = 1
CHOOSE_DURATION = 2
CHOOSE_SCENES = 3
CONFIRM = 4


# --------------------------------------------------------------------------- #
# Preset registry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Preset:
    """User-facing preset. Applied on top of the default config when the
    matching command or inline button fires."""

    label: str
    style_preset_override: str | None = None
    fast_preview: bool = False
    overrides: dict[str, object] = field(default_factory=dict)


PRESETS: dict[str, Preset] = {
    "default": Preset(label="Default / auto"),
    "anime": Preset(label="Anime / manga", style_preset_override="anime"),
    "cinematic": Preset(label="Cinematic photo", style_preset_override="cinematic_photo"),
    "photo": Preset(label="Photoreal", style_preset_override="photoreal"),
    "illustration": Preset(label="Illustration / 3D", style_preset_override="illustration"),
    "fast": Preset(label="Fast preview (3 scenes × 2 s)", fast_preview=True),
}

# The preset picker shown by ``/generate``. Order matters — the two
# rows of three below are laid out as the inline keyboard.
PRESET_PICKER_ORDER: tuple[str, ...] = (
    "cinematic", "photo", "anime",
    "illustration", "default", "fast",
)

# Duration picker: 3-20 s in handy steps. ``None`` means "Qwen decides".
DURATION_CHOICES: tuple[tuple[str, float | None], ...] = (
    ("3 s", 3.0),
    ("5 s", 5.0),
    ("8 s", 8.0),
    ("10 s", 10.0),
    ("15 s", 15.0),
    ("20 s", 20.0),
    ("Auto", None),
)

# Scene-count picker. ``None`` = let Qwen choose 3-6.
SCENES_CHOICES: tuple[tuple[str, int | None], ...] = (
    ("3", 3),
    ("4", 4),
    ("5", 5),
    ("6", 6),
    ("Auto", None),
)


def _build_config(
    run_dir: Path,
    preset: Preset,
    total_duration: float | None = None,
    scenes_count: int | None = None,
) -> PipelineConfig:
    """Build a ``PipelineConfig`` for a preset plus the wizard-collected
    total duration and scene count. Tolerates older ``PipelineConfig``
    versions missing some fields (e.g. if the bot is checked out ahead
    of the orchestrator) by silently skipping unknown kwargs."""
    base_kwargs: dict[str, object] = {
        "comfyui_url": COMFYUI_URL,
        "output_dir": run_dir,
    }
    optional_kwargs: dict[str, object] = {
        "style_preset_override": preset.style_preset_override,
        "fast_preview": preset.fast_preview,
        "total_duration_hint": total_duration,
        "scenes_count_hint": scenes_count,
    }
    optional_kwargs.update(preset.overrides)
    field_names = {f.name for f in PipelineConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    for k, v in optional_kwargs.items():
        if k in field_names:
            base_kwargs[k] = v
        else:
            log.warning("PipelineConfig has no field %r; skipping for preset %r", k, preset.label)
    return PipelineConfig(**base_kwargs)


# --------------------------------------------------------------------------- #
# Google Drive helper (optional fallback)
# --------------------------------------------------------------------------- #


def _drive_service():
    """Lazily build a Drive service if credentials.json is present."""
    if not Path(CREDENTIALS_FILE).is_file():
        return None
    from google.oauth2 import service_account
    from googleapiclient.discovery import build

    credentials = service_account.Credentials.from_service_account_file(
        CREDENTIALS_FILE,
        scopes=["https://www.googleapis.com/auth/drive.file"],
    )
    return build("drive", "v3", credentials=credentials)


def upload_file_to_drive(file_path: Path) -> str:
    svc = _drive_service()
    if svc is None:
        raise RuntimeError(
            "Google Drive credentials not configured; set GOOGLE_APPLICATION_CREDENTIALS "
            "or place credentials.json next to bot.py to enable the fallback upload."
        )
    from googleapiclient.http import MediaFileUpload

    metadata: dict[str, object] = {"name": file_path.name}
    if DRIVE_FOLDER_ID:
        metadata["parents"] = [DRIVE_FOLDER_ID]
    media = MediaFileUpload(str(file_path), resumable=True)
    created = svc.files().create(body=metadata, media_body=media, fields="id").execute()
    svc.permissions().create(
        fileId=created["id"],
        body={"type": "anyone", "role": "reader"},
    ).execute()
    return f"https://drive.google.com/uc?id={created['id']}&export=download"


# --------------------------------------------------------------------------- #
# ComfyUI pipeline wrapper
# --------------------------------------------------------------------------- #


async def generate_video(
    prompt_text: str,
    preset: Preset,
    request_id: str | int = 0,
    total_duration: float | None = None,
    scenes_count: int | None = None,
) -> Path:
    """Run the end-to-end ScenePipeline and return the final mp4 path."""
    run_dir = OUTPUT_DIR / f"run_{request_id}"
    config = _build_config(
        run_dir, preset,
        total_duration=total_duration,
        scenes_count=scenes_count,
    )

    def _run() -> Path:
        pipeline = ScenePipeline(config)
        result = pipeline.run(prompt_text)
        if result.final_video_path is None:
            raise PipelineError(
                "Pipeline finished but no final video was produced; "
                "check ffmpeg availability and per-scene artifacts."
            )
        return result.final_video_path

    return await asyncio.to_thread(_run)


# --------------------------------------------------------------------------- #
# Inline keyboard builders
# --------------------------------------------------------------------------- #


def _preset_keyboard() -> InlineKeyboardMarkup:
    """Two rows of three preset buttons, in ``PRESET_PICKER_ORDER``."""
    buttons: list[list[InlineKeyboardButton]] = []
    row: list[InlineKeyboardButton] = []
    for i, key in enumerate(PRESET_PICKER_ORDER):
        preset = PRESETS[key]
        row.append(InlineKeyboardButton(preset.label, callback_data=f"preset:{key}"))
        if (i + 1) % 3 == 0:
            buttons.append(row)
            row = []
    if row:
        buttons.append(row)
    return InlineKeyboardMarkup(buttons)


def _duration_keyboard() -> InlineKeyboardMarkup:
    """Two rows: six numeric durations + Auto."""
    buttons: list[list[InlineKeyboardButton]] = [[], []]
    for i, (label, value) in enumerate(DURATION_CHOICES):
        token = "auto" if value is None else f"{value:g}"
        buttons[0 if i < 4 else 1].append(
            InlineKeyboardButton(label, callback_data=f"dur:{token}")
        )
    return InlineKeyboardMarkup([row for row in buttons if row])


def _scenes_keyboard() -> InlineKeyboardMarkup:
    row = [
        InlineKeyboardButton(
            label, callback_data=f"scenes:{'auto' if val is None else val}"
        )
        for label, val in SCENES_CHOICES
    ]
    return InlineKeyboardMarkup([row])


def _confirm_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ Старт", callback_data="confirm:yes"),
            InlineKeyboardButton("✖️ Отмена", callback_data="confirm:no"),
        ]
    ])


# --------------------------------------------------------------------------- #
# Conversation helpers
# --------------------------------------------------------------------------- #


def _preset_key_from_command(command: str) -> str:
    """Map the raw command text to a preset key. Used by both the
    command dispatcher and the unit tests.

    ``/start`` / ``/generate`` -> ``"default"`` (wizard chooses preset).
    ``/generate_anime`` -> ``"anime"`` (skip preset picker).
    Unknown suffixes fall back to ``"default"``.
    """
    if not command.startswith("/"):
        return "default"
    head = command.split()[0].lstrip("/").split("@", 1)[0].lower()
    if head in ("start", "generate"):
        return "default"
    if head.startswith("generate_"):
        suffix = head[len("generate_"):]
        if suffix in PRESETS:
            return suffix
    return "default"


def _inline_idea(text: str) -> str:
    """``/generate_anime ninja cat`` -> ``"ninja cat"``; non-command
    text passes through; bare command returns ``""``."""
    text = (text or "").strip()
    if not text.startswith("/"):
        return text
    parts = text.split(maxsplit=1)
    return parts[1].strip() if len(parts) > 1 else ""


def _summary(user_data: dict) -> str:
    preset: Preset = user_data["preset"]
    idea: str = user_data["idea"]
    total: float | None = user_data.get("total_duration")
    scenes: int | None = user_data.get("scenes_count")
    dur_s = f"{total:g} s" if total else "auto"
    scenes_s = str(scenes) if scenes else "auto"
    return (
        f"Пресет: *{preset.label}*\n"
        f"Идея: _{idea}_\n"
        f"Длина: *{dur_s}*\n"
        f"Сцен: *{scenes_s}*"
    )


# --------------------------------------------------------------------------- #
# Conversation handlers
# --------------------------------------------------------------------------- #


async def _command_entry(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Entry point for ``/start``, ``/generate`` and ``/generate_*``."""
    text = (update.message.text or "") if update.message else ""
    preset_key = _preset_key_from_command(text)
    preset = PRESETS[preset_key]
    context.user_data.clear()
    context.user_data["preset_key"] = preset_key
    context.user_data["preset"] = preset

    inline_idea = _inline_idea(text)
    if inline_idea:
        context.user_data["idea"] = inline_idea
        return await _after_idea(update, context)

    if preset_key == "default":
        # Full wizard — start with preset picker.
        await update.message.reply_text(
            "Выбери пресет визуала:",
            reply_markup=_preset_keyboard(),
        )
        return CHOOSE_PRESET

    # Shortcut command (/generate_anime etc.) — preset already locked,
    # ask for the idea directly.
    await update.message.reply_text(
        f"Пресет: *{preset.label}*\nОпиши идею видео одним сообщением.",
        parse_mode="Markdown",
    )
    return AWAIT_IDEA


async def _handle_preset_choice(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> int:
    """CallbackQuery from the preset inline keyboard."""
    query = update.callback_query
    await query.answer()
    _, key = (query.data or "").split(":", 1)
    preset = PRESETS.get(key, PRESETS["default"])
    context.user_data["preset_key"] = key
    context.user_data["preset"] = preset
    await query.edit_message_text(
        f"Пресет: *{preset.label}*\nТеперь опиши идею одним сообщением.",
        parse_mode="Markdown",
    )
    return AWAIT_IDEA


async def _handle_idea_message(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> int:
    """User typed the idea text — persist and move to the next step."""
    idea = (update.message.text or "").strip()
    if not idea:
        await update.message.reply_text(
            "Идея не распозналась. Попробуй ещё раз или /cancel."
        )
        return AWAIT_IDEA
    context.user_data["idea"] = idea
    return await _after_idea(update, context)


async def _after_idea(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> int:
    """Next step after we have ``preset`` + ``idea`` in ``user_data``.
    Fast-preview skips duration/scenes pickers (hardcoded by the
    preset); other presets ask for both."""
    preset: Preset = context.user_data["preset"]
    if preset.fast_preview:
        # Fast preview has its own per-scene duration / count hardcoded.
        return await _ask_confirm(update, context)
    await (update.effective_message or update.message).reply_text(
        "Какая длина ролика целиком?",
        reply_markup=_duration_keyboard(),
    )
    return CHOOSE_DURATION


async def _handle_duration_choice(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> int:
    query = update.callback_query
    await query.answer()
    _, token = (query.data or "").split(":", 1)
    context.user_data["total_duration"] = None if token == "auto" else float(token)
    await query.edit_message_text(
        f"Длина: *{token} s*" if token != "auto" else "Длина: *auto*",
        parse_mode="Markdown",
    )
    await query.message.reply_text(
        "Сколько сцен?",
        reply_markup=_scenes_keyboard(),
    )
    return CHOOSE_SCENES


async def _handle_scenes_choice(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> int:
    query = update.callback_query
    await query.answer()
    _, token = (query.data or "").split(":", 1)
    context.user_data["scenes_count"] = None if token == "auto" else int(token)
    return await _ask_confirm(update, context)


async def _ask_confirm(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> int:
    msg = update.effective_message or update.message
    await msg.reply_text(
        _summary(context.user_data) + "\n\nЗапускаем?",
        parse_mode="Markdown",
        reply_markup=_confirm_keyboard(),
    )
    return CONFIRM


async def _handle_confirm(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> int:
    query = update.callback_query
    await query.answer()
    _, choice = (query.data or "").split(":", 1)
    if choice != "yes":
        await query.edit_message_text("Отменено.")
        return ConversationHandler.END
    await query.edit_message_text(_summary(context.user_data) + "\n\nГенерирую…")
    await _run_and_reply(update, context)
    return ConversationHandler.END


async def _run_and_reply(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    preset: Preset = context.user_data["preset"]
    idea: str = context.user_data["idea"]
    total = context.user_data.get("total_duration")
    scenes = context.user_data.get("scenes_count")
    chat_id = update.effective_chat.id
    reply_to = update.effective_message

    try:
        video_path = await generate_video(
            idea, preset,
            request_id=chat_id,
            total_duration=total,
            scenes_count=scenes,
        )
    except PipelineError as exc:
        await reply_to.reply_text(f"Ошибка генерации: {exc}")
        return
    except Exception as exc:  # noqa: BLE001
        log.exception("Unhandled pipeline error")
        await reply_to.reply_text(f"Неожиданная ошибка: {exc}")
        return

    size_mb = video_path.stat().st_size / (1024 * 1024)
    try:
        if size_mb <= TELEGRAM_MAX_UPLOAD_MB:
            with video_path.open("rb") as vf:
                await reply_to.reply_video(video=vf)
        else:
            raise RuntimeError(
                f"video {size_mb:.1f} MB > Telegram limit, uploading to Drive"
            )
    except Exception as exc:  # noqa: BLE001
        log.warning("Direct video send failed, falling back to Drive: %s", exc)
        try:
            url = upload_file_to_drive(video_path)
            await reply_to.reply_text(f"Готово! Видео: {url}")
        except Exception as drive_exc:  # noqa: BLE001
            await reply_to.reply_text(
                f"Видео готово локально ({video_path}), но отправить в чат не удалось: "
                f"{drive_exc}"
            )


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    msg = update.effective_message or update.message
    if msg is not None:
        await msg.reply_text("Диалог завершён.")
    return ConversationHandler.END


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    lines = [
        "*Команды*",
        "/generate — мастер: пресет → идея → длина → сцены → подтверждение",
        "/generate\\_anime — аниме / манга (пропускает выбор пресета)",
        "/generate\\_cinematic — кинематографичное фото",
        "/generate\\_photo — фотореализм (RealVis)",
        "/generate\\_illustration — 3D / иллюстрация",
        "/generate\\_fast — быстрый превью-режим (3 сцены × 2 сек, без выбора длины)",
        "/cancel — прервать диалог",
        "",
        "Идею можно передать сразу после команды одним сообщением, например:",
        "`/generate_anime ninja cat in rainy Tokyo`",
    ]
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


# --------------------------------------------------------------------------- #
# Wiring
# --------------------------------------------------------------------------- #


def _build_conv_handler() -> ConversationHandler:
    entry_cmds = ["start", "generate"] + [
        f"generate_{k}" for k in PRESETS if k != "default"
    ]
    entry_points = [CommandHandler(name, _command_entry) for name in entry_cmds]
    return ConversationHandler(
        entry_points=entry_points,
        states={
            CHOOSE_PRESET: [
                CallbackQueryHandler(_handle_preset_choice, pattern=r"^preset:"),
            ],
            AWAIT_IDEA: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, _handle_idea_message),
            ],
            CHOOSE_DURATION: [
                CallbackQueryHandler(_handle_duration_choice, pattern=r"^dur:"),
            ],
            CHOOSE_SCENES: [
                CallbackQueryHandler(_handle_scenes_choice, pattern=r"^scenes:"),
            ],
            CONFIRM: [
                CallbackQueryHandler(_handle_confirm, pattern=r"^confirm:"),
            ],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )


def main() -> None:
    if not API_KEY:
        raise SystemExit(
            "TELEGRAM_BOT_TOKEN is not set. Export it or configure it via Devin's "
            "secrets manager before starting the bot."
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    app = ApplicationBuilder().token(API_KEY).build()
    app.add_handler(_build_conv_handler())
    app.add_handler(CommandHandler("help", help_command))

    log.info(
        "Бот запущен, COMFYUI_URL=%s output=%s presets=%s",
        COMFYUI_URL, OUTPUT_DIR, list(PRESETS.keys()),
    )
    app.run_polling()


if __name__ == "__main__":
    main()
