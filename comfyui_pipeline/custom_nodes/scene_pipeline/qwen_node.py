"""ComfyUI node: run a local Qwen model to generate a JSON scenario.

Uses Hugging Face transformers. The model is loaded lazily on first use and
kept in memory between calls (unless ``keep_loaded=False``). Accepts any
Qwen-family causal LM identifier (e.g. ``Qwen/Qwen2.5-7B-Instruct``) or a
local path.
"""
from __future__ import annotations

import gc
import logging
import threading
from typing import Any

log = logging.getLogger(__name__)

_MODEL_CACHE: dict[str, Any] = {}
_CACHE_LOCK = threading.Lock()


def _load_model(model_name_or_path: str, device: str, dtype: str):
    """Load model+tokenizer with a simple process-local cache."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    key = f"{model_name_or_path}|{device}|{dtype}"
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached

    torch_dtype: Any
    if dtype == "auto":
        torch_dtype = "auto"
    elif dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif dtype == "fp16":
        torch_dtype = torch.float16
    elif dtype == "fp32":
        torch_dtype = torch.float32
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    resolved_device = device
    if resolved_device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"

    log.info("Loading Qwen model %s on %s (dtype=%s)", model_name_or_path, resolved_device, dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        device_map=resolved_device if resolved_device != "auto" else "auto",
        trust_remote_code=True,
    )
    model.eval()
    _MODEL_CACHE[key] = (tokenizer, model, resolved_device)
    return _MODEL_CACHE[key]


def _unload_all() -> None:
    import torch

    _MODEL_CACHE.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class QwenScenarioGenerator:
    """ComfyUI node wrapper that runs Qwen locally and returns a string.

    The node is marked as an OUTPUT_NODE so that its generated text is stored
    inside ``/history/{prompt_id}.outputs[<node_id>].text`` and can be read by
    the orchestrator.
    """

    CATEGORY = "scene_pipeline"
    FUNCTION = "generate"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("scenario_json",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "model_name_or_path": (
                    "STRING",
                    {"default": "Qwen/Qwen3-8B", "multiline": False},
                ),
                "system_prompt": ("STRING", {"multiline": True, "default": ""}),
                "user_prompt": ("STRING", {"multiline": True, "default": ""}),
                "max_new_tokens": ("INT", {"default": 2048, "min": 64, "max": 16384}),
                "temperature": (
                    "FLOAT",
                    {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05},
                ),
                "top_p": (
                    "FLOAT",
                    {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "dtype": (["auto", "bf16", "fp16", "fp32"], {"default": "auto"}),
                "keep_loaded": ("BOOLEAN", {"default": True}),
            }
        }

    def generate(
        self,
        model_name_or_path: str,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
        device: str,
        dtype: str,
        keep_loaded: bool,
    ) -> dict[str, Any]:
        import torch

        with _CACHE_LOCK:
            tokenizer, model, resolved_device = _load_model(model_name_or_path, device, dtype)

        messages = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": user_prompt.strip()})

        template_kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        # Qwen3 supports a "thinking mode" that emits <think>...</think>. For
        # strict JSON output we disable it when the tokenizer accepts the flag.
        try:
            prompt_text = tokenizer.apply_chat_template(
                messages, enable_thinking=False, **template_kwargs
            )
        except TypeError:
            prompt_text = tokenizer.apply_chat_template(messages, **template_kwargs)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(resolved_device)

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        do_sample = temperature > 0.0
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
                top_p=top_p if do_sample else 1.0,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        # Strip any leading <think>...</think> block produced by Qwen3 in
        # thinking mode (older tokenizers ignore enable_thinking=False).
        import re as _re
        text = _re.sub(r"^<think>.*?</think>\s*", "", text, flags=_re.DOTALL).strip()

        if not keep_loaded:
            _unload_all()

        return {"ui": {"text": [text]}, "result": (text,)}


__all__ = ["QwenScenarioGenerator"]
