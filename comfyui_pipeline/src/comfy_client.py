"""Minimal ComfyUI HTTP+WebSocket client used by the orchestrator."""
from __future__ import annotations

import json
import logging
import time
import urllib.parse
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
import websocket

log = logging.getLogger(__name__)


@dataclass
class ComfyOutputFile:
    """Descriptor of a file produced by ComfyUI."""

    filename: str
    subfolder: str
    type: str  # "output" | "temp" | "input"


class ComfyClient:
    """Thin wrapper around the ComfyUI /prompt, /history, /view and /ws endpoints.

    Only covers what the orchestrator needs: queue a workflow, wait for it to
    finish via websocket events, fetch the resulting files and text outputs.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8188", client_id: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.client_id = client_id or str(uuid.uuid4())
        parsed = urllib.parse.urlparse(self.base_url)
        ws_scheme = "wss" if parsed.scheme == "https" else "ws"
        self._ws_url = f"{ws_scheme}://{parsed.netloc}/ws?clientId={self.client_id}"

    def queue_prompt(self, workflow: dict[str, Any]) -> str:
        """POST the workflow (API format) to /prompt and return the prompt_id."""
        resp = requests.post(
            f"{self.base_url}/prompt",
            json={"prompt": workflow, "client_id": self.client_id},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if "prompt_id" not in data:
            raise RuntimeError(f"ComfyUI did not return prompt_id: {data}")
        return data["prompt_id"]

    def wait_for_completion(self, prompt_id: str, timeout: float = 3600.0) -> None:
        """Block on the websocket until the given prompt_id finishes executing."""
        ws = websocket.WebSocket()
        ws.connect(self._ws_url, timeout=30)
        ws.settimeout(timeout)
        deadline = time.monotonic() + timeout
        try:
            while time.monotonic() < deadline:
                raw = ws.recv()
                if not isinstance(raw, str):
                    # Binary frames are preview images — ignore.
                    continue
                msg = json.loads(raw)
                mtype = msg.get("type")
                mdata = msg.get("data", {})
                if mdata.get("prompt_id") != prompt_id:
                    continue
                if mtype == "executing" and mdata.get("node") is None:
                    return
                if mtype == "execution_error":
                    raise RuntimeError(f"ComfyUI execution error: {mdata}")
                if mtype == "execution_cached" and mdata.get("nodes") is None:
                    return
            raise TimeoutError(f"Timed out waiting for prompt {prompt_id}")
        finally:
            ws.close()

    def get_history(self, prompt_id: str) -> dict[str, Any]:
        resp = requests.get(f"{self.base_url}/history/{prompt_id}", timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        if prompt_id not in payload:
            raise RuntimeError(f"No history for prompt {prompt_id}")
        return payload[prompt_id]

    def run(self, workflow: dict[str, Any], timeout: float = 3600.0) -> dict[str, Any]:
        """Queue workflow, wait for completion, return history entry."""
        prompt_id = self.queue_prompt(workflow)
        log.info("Queued prompt %s", prompt_id)
        self.wait_for_completion(prompt_id, timeout=timeout)
        return self.get_history(prompt_id)

    def fetch_file(self, file: ComfyOutputFile, dest: Path) -> Path:
        """Download one output/temp file via /view to dest."""
        params = {"filename": file.filename, "subfolder": file.subfolder, "type": file.type}
        resp = requests.get(f"{self.base_url}/view", params=params, timeout=300, stream=True)
        resp.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 16):
                if chunk:
                    fh.write(chunk)
        return dest

    @staticmethod
    def collect_outputs(history: dict[str, Any]) -> dict[str, list[ComfyOutputFile]]:
        """Group output file descriptors by node_id, for any output key."""
        result: dict[str, list[ComfyOutputFile]] = {}
        for node_id, node_out in history.get("outputs", {}).items():
            files: list[ComfyOutputFile] = []
            for key in ("images", "gifs", "videos", "audio"):
                for item in node_out.get(key, []) or []:
                    files.append(
                        ComfyOutputFile(
                            filename=item["filename"],
                            subfolder=item.get("subfolder", ""),
                            type=item.get("type", "output"),
                        )
                    )
            if files:
                result[node_id] = files
        return result

    @staticmethod
    def collect_texts(history: dict[str, Any]) -> dict[str, list[str]]:
        """Collect any text/string UI outputs per node_id.

        Supports nodes that expose text via `outputs[node].text` (list of strings)
        or `outputs[node].string` (list of strings). This matches the conventions
        used by ShowText/SaveText-style nodes and the pipeline's own QwenScenarioGenerator.
        """
        result: dict[str, list[str]] = {}
        for node_id, node_out in history.get("outputs", {}).items():
            texts: list[str] = []
            for key in ("text", "string"):
                val = node_out.get(key)
                if isinstance(val, list):
                    texts.extend(str(x) for x in val)
                elif isinstance(val, str):
                    texts.append(val)
            if texts:
                result[node_id] = texts
        return result
