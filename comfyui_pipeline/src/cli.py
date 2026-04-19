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

# Pull defaults from PipelineConfig so CLI flags and programmatic use stay in sync.
_DEFAULTS = PipelineConfig()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a multi-scene video from a text idea via ComfyUI (Qwen → SDXL → LTX)."
    )
    parser.add_argument("idea", help="The text idea describing the video you want")
    parser.add_argument(
        "--comfyui-url",
        default=os.environ.get("COMFYUI_URL", _DEFAULTS.comfyui_url),
        help="ComfyUI HTTP endpoint (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("SCENE_OUTPUT_DIR", str(_DEFAULTS.output_dir))),
    )
    parser.add_argument("--qwen-model", default=_DEFAULTS.qwen_model)
    parser.add_argument(
        "--qwen-device",
        choices=("auto", "cuda", "cpu"),
        default=_DEFAULTS.qwen_device,
        help=(
            "Where Qwen runs. 'auto' lets accelerate split layers across "
            "CPU+GPU (recommended for <16 GB VRAM); 'cpu' forces CPU (safest "
            "on 12 GB cards, ~1-2 min per scenario); 'cuda' pins to GPU."
        ),
    )
    qwen_keep = parser.add_mutually_exclusive_group()
    qwen_keep.add_argument(
        "--keep-qwen-loaded",
        dest="qwen_keep_loaded",
        action="store_true",
        help="Keep Qwen resident in memory between runs (fast, needs >=32 GB RAM).",
    )
    qwen_keep.add_argument(
        "--no-keep-qwen-loaded",
        dest="qwen_keep_loaded",
        action="store_false",
        help="Unload Qwen after each scenario (default — frees RAM/VRAM for SDXL+LTX).",
    )
    parser.set_defaults(qwen_keep_loaded=_DEFAULTS.qwen_keep_loaded)
    parser.add_argument("--sdxl-checkpoint", default=_DEFAULTS.sdxl_checkpoint)
    parser.add_argument("--ltx-checkpoint", default=_DEFAULTS.ltx_checkpoint)
    parser.add_argument("--ltx-lora", default=_DEFAULTS.ltx_lora)
    parser.add_argument("--ltx-text-encoder", default=_DEFAULTS.ltx_text_encoder)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--video-fps", type=float, default=_DEFAULTS.video_fps)
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
        qwen_device=args.qwen_device,
        qwen_keep_loaded=args.qwen_keep_loaded,
        sdxl_checkpoint=args.sdxl_checkpoint,
        ltx_checkpoint=args.ltx_checkpoint,
        ltx_lora=args.ltx_lora,
        ltx_text_encoder=args.ltx_text_encoder,
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
