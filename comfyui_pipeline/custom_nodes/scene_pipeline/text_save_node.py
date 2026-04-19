"""ComfyUI node: save a STRING to a text file in the ComfyUI output directory."""
from __future__ import annotations

import os
import re
from typing import Any


def _next_counter(out_dir: str, prefix: str) -> int:
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)\.txt$")
    highest = 0
    if os.path.isdir(out_dir):
        for name in os.listdir(out_dir):
            match = pattern.match(name)
            if match:
                highest = max(highest, int(match.group(1)))
    return highest + 1


class SaveTextToFile:
    """Persist a STRING input to ``{output_dir}/{filename_prefix}_{NNNNN}.txt``.

    Also echoes the text back via ``ui.text`` so it appears in history.
    """

    CATEGORY = "scene_pipeline"
    FUNCTION = "save"
    RETURN_TYPES = ()
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
                "filename_prefix": ("STRING", {"default": "scenario"}),
            }
        }

    def save(self, text: str, filename_prefix: str) -> dict[str, Any]:
        try:
            import folder_paths  # Provided by ComfyUI at runtime.

            out_dir = folder_paths.get_output_directory()
        except Exception:
            out_dir = os.path.abspath("output")
            os.makedirs(out_dir, exist_ok=True)

        counter = _next_counter(out_dir, filename_prefix)
        filename = f"{filename_prefix}_{counter:05d}.txt"
        path = os.path.join(out_dir, filename)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)

        return {"ui": {"text": [text], "saved_path": [path]}}


__all__ = ["SaveTextToFile"]
