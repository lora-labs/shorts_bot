"""ComfyUI custom nodes for the scene pipeline.

Install by symlinking or copying this directory into
`ComfyUI/custom_nodes/scene_pipeline`.
"""
from .qwen_node import QwenScenarioGenerator
from .text_save_node import SaveTextToFile

NODE_CLASS_MAPPINGS = {
    "QwenScenarioGenerator": QwenScenarioGenerator,
    "SaveTextToFile": SaveTextToFile,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenScenarioGenerator": "Qwen Scenario Generator",
    "SaveTextToFile": "Save Text To File",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
