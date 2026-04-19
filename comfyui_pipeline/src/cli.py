"""Command-line entrypoint for the scene pipeline.

Usage:
    python -m comfyui_pipeline.src.cli "a robot learning to paint" \
        --comfyui-url http://127.0.0.1:8188 \
        --output-dir output/scenes
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from .orchestrator import PipelineConfig, ScenePipeline


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a multi-scene video from a text idea via ComfyUI (Qwen → SDXL → LTX)."
    )
    parser.add_argument("idea", help="The text idea describing the video you want")
    parser.add_argument(
        "--comfyui-url",
        default=os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188"),
        help="ComfyUI HTTP endpoint (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("SCENE_OUTPUT_DIR", "output/scenes")),
    )
    parser.add_argument("--qwen-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--sdxl-checkpoint", default="sd_xl_base_1.0.safetensors")
    parser.add_argument("--ltx-checkpoint", default="ltx-video-2b-v0.9.safetensors")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--video-fps", type=float, default=25.0)
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = PipelineConfig(
        comfyui_url=args.comfyui_url,
        output_dir=args.output_dir,
        qwen_model=args.qwen_model,
        sdxl_checkpoint=args.sdxl_checkpoint,
        ltx_checkpoint=args.ltx_checkpoint,
        seed=args.seed,
        video_fps=args.video_fps,
    )
    pipeline = ScenePipeline(cfg)
    result = pipeline.run(args.idea)

    print(f"\nTitle: {result.scenario.title}")
    print(f"Scenes: {len(result.scene_artifacts)}")
    for sa in result.scene_artifacts:
        print(f"  [{sa.scene.id}] {sa.video_path}")
    print(f"\nAll artifacts: {result.output_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
