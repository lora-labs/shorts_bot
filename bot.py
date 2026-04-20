"""Telegram bot entry point.

Commands
--------
* ``/start`` — legacy entry point. Asks for an idea, uses default preset.
* ``/generate`` — same as /start but newer wording.
* ``/generate_anime`` — forces the anime SDXL checkpoint.
* ``/generate_cinematic`` — forces the cinematic photoreal checkpoint.
* ``/generate_photo`` — forces the alternate photoreal (RealVisXL) checkpoint.
* ``/generate_illustration`` — forces the stylized illustration checkpoint.
* ``/generate_fast`` — fast-preview mode (3 scenes × 2 s each). Useful to
  iterate on an idea before committing to a 30-minute full render.
* ``/cancel`` — exit the conversation.

Each command accepts the idea either inline (``/generate_anime a ninja cat
in Tokyo``) or in a follow-up message. Idle or malformed input times out
and the conversation ends politely.

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

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
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

AWAIT_IDEA = 0


# --------------------------------------------------------------------------- #
# Preset registry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Preset:
    """User-facing command preset. Applied on top of the default config
    when the matching ``/generate_*`` command fires."""

    label: str
    style_preset_override: str | None = None
    fast_preview: bool = False
    overrides: dict[str, object] = field(default_factory=dict)


# Telegram command suffix -> Preset. The suffix is also the identifier
# stored in user_data so post-command message handler knows which config
# to apply.
PRESETS: dict[str, Preset] = {
    "default": Preset(label="Default"),
    "anime": Preset(
        label="Anime / manga",
        style_preset_override="anime",
    ),
    "cinematic": Preset(
        label="Cinematic photo",
        style_preset_override="cinematic_photo",
    ),
    "photo": Preset(
        label="Photoreal",
        style_preset_override="photoreal",
    ),
    "illustration": Preset(
        label="Illustration / 3D",
        style_preset_override="illustration",
    ),
    "fast": Preset(
        label="Fast preview (3 scenes × 2 s)",
        fast_preview=True,
    ),
}


def _build_config(run_dir: Path, preset: Preset) -> PipelineConfig:
    """Build a ``PipelineConfig`` for a preset, tolerating the case where
    the PR #16 / #17 fields are not yet present in ``PipelineConfig``
    (e.g. if someone runs this bot against an older branch) — unknown
    fields are silently skipped instead of crashing at startup.
    """
    base_kwargs: dict[str, object] = {
        "comfyui_url": COMFYUI_URL,
        "output_dir": run_dir,
    }
    optional_kwargs: dict[str, object] = {
        "style_preset_override": preset.style_preset_override,
        "fast_preview": preset.fast_preview,
    }
    optional_kwargs.update(preset.overrides)
    # Keep only fields PipelineConfig actually declares.
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
) -> Path:
    """Run the end-to-end ScenePipeline and return the final mp4 path."""
    run_dir = OUTPUT_DIR / f"run_{request_id}"
    config = _build_config(run_dir, preset)

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
# Telegram handlers
# --------------------------------------------------------------------------- #


def _preset_from_update(update: Update) -> Preset:
    """Derive the active preset from the triggering command name.
    ``/generate_anime`` -> PRESETS['anime']; unknown or legacy /start -> default.
    """
    text = (update.message.text or "").strip() if update.message else ""
    if not text.startswith("/"):
        return PRESETS["default"]
    cmd = text.split()[0].lstrip("/").split("@", 1)[0].lower()
    if cmd in ("start", "generate"):
        return PRESETS["default"]
    if cmd.startswith("generate_"):
        key = cmd[len("generate_"):]
        return PRESETS.get(key, PRESETS["default"])
    return PRESETS["default"]


def _inline_idea(update: Update) -> str:
    """Extract the idea when the user passes it inline with the command:
    ``/generate_anime a ninja cat in Tokyo`` -> ``"a ninja cat in Tokyo"``.
    Returns empty string when nothing follows the command.
    """
    text = (update.message.text or "").strip() if update.message else ""
    if not text.startswith("/"):
        return text
    parts = text.split(maxsplit=1)
    return parts[1].strip() if len(parts) > 1 else ""


async def _command_entry(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Unified entry for all /generate* and /start commands."""
    preset = _preset_from_update(update)
    context.user_data["preset_key"] = next(
        (k for k, v in PRESETS.items() if v is preset), "default"
    )

    inline_idea = _inline_idea(update)
    if inline_idea:
        # User passed the idea inline — skip the follow-up prompt.
        await _run_and_reply(update, context, inline_idea, preset)
        return ConversationHandler.END

    await update.message.reply_text(
        f"Пресет: *{preset.label}*\n"
        "Опиши идею видео одним сообщением — я сгенерирую сценарий, "
        "отрендерю кадры и пришлю готовый ролик.",
        parse_mode="Markdown",
    )
    return AWAIT_IDEA


async def _handle_idea_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Second turn of the conversation: user sent the idea text."""
    preset_key = context.user_data.get("preset_key", "default")
    preset = PRESETS.get(preset_key, PRESETS["default"])
    idea = (update.message.text or "").strip()
    if not idea:
        await update.message.reply_text("Идея не распозналась. Попробуй ещё раз или /cancel.")
        return AWAIT_IDEA
    await _run_and_reply(update, context, idea, preset)
    return ConversationHandler.END


async def _run_and_reply(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    idea: str,
    preset: Preset,
) -> None:
    await update.message.reply_text(
        f"Принято (пресет: {preset.label}). Генерирую сценарий, кадры и видео — "
        "это может занять несколько минут."
    )

    try:
        video_path = await generate_video(
            idea, preset, request_id=update.effective_chat.id
        )
    except PipelineError as exc:
        await update.message.reply_text(f"Ошибка генерации: {exc}")
        return
    except Exception as exc:  # noqa: BLE001
        log.exception("Unhandled pipeline error")
        await update.message.reply_text(f"Неожиданная ошибка: {exc}")
        return

    size_mb = video_path.stat().st_size / (1024 * 1024)
    try:
        if size_mb <= TELEGRAM_MAX_UPLOAD_MB:
            with video_path.open("rb") as vf:
                await update.message.reply_video(video=vf)
        else:
            raise RuntimeError(
                f"video {size_mb:.1f} MB > Telegram limit, uploading to Drive"
            )
    except Exception as exc:  # noqa: BLE001
        log.warning("Direct video send failed, falling back to Drive: %s", exc)
        try:
            url = upload_file_to_drive(video_path)
            await update.message.reply_text(f"Готово! Видео: {url}")
        except Exception as drive_exc:  # noqa: BLE001
            await update.message.reply_text(
                f"Видео готово локально ({video_path}), но отправить в чат не удалось: "
                f"{drive_exc}"
            )


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("Диалог завершён.")
    return ConversationHandler.END


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    lines = [
        "*Команды*",
        "/generate — обычная генерация (автовыбор стиля)",
        "/generate\\_anime — аниме / манга",
        "/generate\\_cinematic — кинематографичное фото",
        "/generate\\_photo — фотореализм (RealVis)",
        "/generate\\_illustration — 3D / иллюстрация",
        "/generate\\_fast — быстрый превью-режим (3 сцены × 2 сек)",
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
            AWAIT_IDEA: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, _handle_idea_message),
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
