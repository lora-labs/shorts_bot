"""Gradio Web UI for the ScenePipeline.

Usage
-----
::

    python -m comfyui_pipeline.gradio_app

Environment variables
---------------------
* ``COMFYUI_URL`` — default ``http://127.0.0.1:8188``.
* ``GRADIO_SERVER_NAME`` / ``GRADIO_SERVER_PORT`` / ``GRADIO_SHARE`` — standard
  Gradio launch settings.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import gradio as gr

from comfyui_pipeline.src.orchestrator import (
    PipelineConfig,
    PipelineError,
    ScenePipeline,
)

log = logging.getLogger(__name__)

COMFYUI_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188")
OUTPUT_ROOT = Path(os.environ.get("GRADIO_OUTPUT_DIR", "output/gradio_runs"))


# (image_width, image_height, video_width, video_height)
ASPECT_PRESETS: dict[str, tuple[int, int, int, int]] = {
    "9:16 vertical (Shorts/Reels)": (768, 1344, 544, 960),
    "1:1 square": (1024, 1024, 768, 768),
    "16:9 horizontal": (1344, 768, 960, 544),
}


def _build_config(
    run_dir: Path,
    aspect: str,
    scenes_count: int,
    duration: float,
    ipa_weight: float,
    image_steps: int,
    seed_int: int,
    fast_preview: bool,
    negative_extra: str,
) -> PipelineConfig:
    iw, ih, vw, vh = ASPECT_PRESETS.get(aspect, ASPECT_PRESETS["9:16 vertical (Shorts/Reels)"])
    cfg = PipelineConfig(
        comfyui_url=COMFYUI_URL,
        output_dir=run_dir,
        image_width=iw,
        image_height=ih,
        video_width=vw,
        video_height=vh,
        image_steps=int(image_steps),
        ip_adapter_weight=float(ipa_weight),
        seed=int(seed_int) if seed_int and int(seed_int) > 0 else None,
        fast_preview=bool(fast_preview),
        # 0 on either slider means "no preference — let Qwen pick".
        scenes_count_hint=int(scenes_count) if scenes_count and int(scenes_count) > 0 else None,
        scene_duration_hint=float(duration) if duration and float(duration) > 0 else None,
        negative_prompt_override=negative_extra.strip(),
    )
    return cfg


def run_pipeline(
    idea: str,
    aspect: str,
    scenes_count: int,
    duration: float,
    ipa_weight: float,
    image_steps: int,
    seed_int: int,
    fast_preview: bool,
    negative_extra: str,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
):
    idea = (idea or "").strip()
    if not idea:
        raise gr.Error("Введите идею видео.")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_ROOT / f"run_{stamp}"
    cfg = _build_config(
        run_dir,
        aspect,
        scenes_count,
        duration,
        ipa_weight,
        image_steps,
        seed_int,
        fast_preview,
        negative_extra,
    )

    progress(0.05, desc="Генерация сценария (Qwen3-8B)…")
    pipeline = ScenePipeline(cfg)
    try:
        result = pipeline.run(idea)
    except PipelineError as exc:
        raise gr.Error(f"Ошибка пайплайна: {exc}") from exc

    progress(0.95, desc="Склейка финального видео…")
    final = result.final_video_path
    scene_videos = [str(sa.video_path) for sa in result.scene_artifacts]
    scene_images = [str(sa.image_path) for sa in result.scene_artifacts]
    title = result.scenario.title
    scenes_md = "\n".join(
        f"- **Сцена {sa.scene.id}** ({sa.scene.duration_seconds:.1f}s) — {sa.scene.description}"
        for sa in result.scene_artifacts
    )
    return (
        title,
        scenes_md,
        str(final) if final else None,
        scene_videos,
        scene_images,
    )


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Shorts Bot — Генератор видео") as demo:
        gr.Markdown(
            "# Shorts Bot — Генератор видео\n"
            "Введи идею — получи сценарий (Qwen3-8B) → "
            "кадры (SDXL + IP-Adapter) → видео (LTX-2.3) → "
            "склееный ролик."
        )
        idea = gr.Textbox(
            label="Идея видео",
            placeholder="Например: «Собака-космонавт исследует неизвестную планету…»",
            lines=3,
        )

        with gr.Accordion("Параметры генерации", open=False):
            with gr.Row():
                aspect = gr.Dropdown(
                    label="Соотношение сторон",
                    choices=list(ASPECT_PRESETS.keys()),
                    value="9:16 vertical (Shorts/Reels)",
                )
                scenes_count = gr.Slider(
                    label="Количество сцен (0 = авто)",
                    minimum=0, maximum=8, step=1, value=0,
                    info="0 — Qwen сам решит по сюжету",
                )
                duration = gr.Slider(
                    label="Длительность сцены, сек (0 = авто)",
                    minimum=0.0, maximum=8.0, step=0.5, value=0.0,
                    info="0 — Qwen сам решит по сюжету",
                )
            with gr.Row():
                ipa_weight = gr.Slider(
                    label="IP-Adapter weight (сила лока персонажа)",
                    minimum=0.0, maximum=1.0, step=0.05, value=0.55,
                    info="0.4 — больше свободы сцен, 0.75 — жёсткий лок идентичности",
                )
                image_steps = gr.Slider(
                    label="SDXL steps",
                    minimum=10, maximum=50, step=1, value=28,
                )
                seed_int = gr.Number(
                    label="Seed (0 = случайный)",
                    value=0, precision=0,
                )
            with gr.Row():
                fast_preview = gr.Checkbox(
                    label="Fast preview (3 сцены × 2 сек, для проверки идеи)",
                    value=False,
                )
            negative_extra = gr.Textbox(
                label="Дополнительный негативный промпт (добавится к каждой сцене)",
                placeholder="text, watermark, extra fingers, deformed…",
                lines=2,
            )

        btn = gr.Button("Сгенерировать", variant="primary")
        with gr.Row():
            title_out = gr.Textbox(label="Название сценария", interactive=False)
        scenes_out = gr.Markdown(label="Сцены")
        final_video = gr.Video(label="Готовое видео (все сцены склеены)")
        with gr.Row():
            scene_gallery = gr.Gallery(label="Ключевые кадры", columns=4, height="auto")
            scene_clips = gr.Files(label="Видео по сценам")
        btn.click(
            run_pipeline,
            inputs=[
                idea, aspect, scenes_count, duration, ipa_weight,
                image_steps, seed_int, fast_preview, negative_extra,
            ],
            outputs=[title_out, scenes_out, final_video, scene_clips, scene_gallery],
        )
    return demo


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    demo = build_demo()
    demo.queue().launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        share=os.environ.get("GRADIO_SHARE", "false").lower() == "true",
    )
