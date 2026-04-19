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


def run_pipeline(idea: str, progress: gr.Progress = gr.Progress(track_tqdm=False)):
    idea = (idea or "").strip()
    if not idea:
        raise gr.Error("Введите идею видео.")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_ROOT / f"run_{stamp}"
    cfg = PipelineConfig(comfyui_url=COMFYUI_URL, output_dir=run_dir)

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
            "кадры (SDXL + KSamplerAdvanced) → видео (LTX-2.3 Distilled) → "
            "склееный ролик."
        )
        with gr.Row():
            idea = gr.Textbox(
                label="Идея видео",
                placeholder="Например: «Собака-космонавт исследует неизвестную планету…»",
                lines=3,
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
            inputs=[idea],
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
