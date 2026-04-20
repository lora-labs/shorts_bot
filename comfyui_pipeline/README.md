# ComfyUI Scene Pipeline

End-to-end pipeline that turns a **text idea** into a finished short video.
One input, one output:

```
text idea
  └─► Qwen3-8B (local)  ──► JSON scenario ──► for each scene:
                                              ├─► SDXL + KSamplerAdvanced (keyframe PNG)
                                              └─► LTX-2.3 Distilled (img → MP4)
                                           ──► ffmpeg concat ──► final.mp4
```

Three entry points wrap the same orchestrator:

* **CLI** — `python -m comfyui_pipeline.src.cli "idea"`
* **Telegram bot** — `python bot.py` (reads `TELEGRAM_BOT_TOKEN`)
* **Web UI** — `python -m comfyui_pipeline.gradio_app` (Gradio)

Two custom ComfyUI nodes (`QwenScenarioGenerator`, `SaveTextToFile`) handle the
LLM stage locally; everything else uses stock ComfyUI + official LTX-2.3 nodes.

## Layout

```
comfyui_pipeline/
├── custom_nodes/scene_pipeline/    # drop-in ComfyUI custom node package
│   ├── qwen_node.py                #   QwenScenarioGenerator (Qwen3-aware)
│   └── text_save_node.py           #   SaveTextToFile
├── workflows/                      # ComfyUI "API format" workflows
│   ├── script_gen_api.json         #   idea → Qwen3 → JSON
│   ├── scene_image_api.json        #   prompt → SDXL + KSamplerAdvanced → PNG
│   └── scene_video_api.json        #   image + prompt → LTX-2.3 → MP4
├── prompts/system_prompt.txt       # JSON-only system prompt for Qwen
├── src/
│   ├── comfy_client.py             # minimal HTTP + WS client
│   ├── orchestrator.py             # pipeline glue + ffmpeg concat
│   ├── schema.py                   # Pydantic Scenario/Scene models
│   └── cli.py                      # `python -m comfyui_pipeline.src.cli ...`
├── gradio_app.py                   # Web UI
├── tests/                          # offline unit tests (no ComfyUI required)
└── requirements.txt
```

The Telegram bot lives at the repo root as `bot.py`.

## Prerequisites

* A running **ComfyUI** with official LTX-2 support (uses core nodes
  `CLIPLoader` (type=`ltxv`), `SamplerCustomAdvanced`, `LTXVConditioning`, etc.).
  Upgrade to the latest ComfyUI and install **ComfyUI-LTXVideo** for the
  custom LTX helper nodes (not strictly required for this pipeline, but handy).
* **FFmpeg** in `PATH` — the orchestrator uses it to concatenate per-scene
  MP4s into `final.mp4`.
* Python 3.10+ on the box running the orchestrator (can be the same host).
* CUDA GPU. VRAM floor for the defaults:
  * SDXL base: ~8 GB
  * **LTX-2.3 22B Distilled**: ~32 GB (use the `ltx-2.3-…-distilled-lora-…`
    LoRA we ship defaults for; see below)
  * **Qwen3-8B** bf16: ~17 GB (or load it 4-bit via `transformers`)
  * Stages run sequentially, so peak VRAM = max(stage), not the sum.

### Required model files

Place the following under your ComfyUI install (paths relative to
`ComfyUI/models/`):

| Path | Source |
| --- | --- |
| `checkpoints/sd_xl_base_1.0.safetensors` | <https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0> |
| `checkpoints/ltx-2.3-22b-distilled-1.1.safetensors` | <https://huggingface.co/Lightricks/LTX-Video> |
| `loras/ltx-2.3-22b-distilled-lora-384-1.1.safetensors` | same Lightricks repo |
| `text_encoders/comfy_gemma_3_12B_it.safetensors` | same Lightricks repo |

Qwen3-8B is pulled on first run via `transformers` (defaults to
`Qwen/Qwen3-8B`). Override with `--qwen-model` or
`PipelineConfig.qwen_model=` (e.g. a local path, or `Qwen/Qwen3-14B` if you
have the VRAM).

On low-VRAM GPUs (e.g. RTX 4070 12 GB) Qwen3-8B in fp16 (~16 GB) does not
fit on the GPU alongside SDXL/LTX. Defaults are tuned for that:

- `--qwen-device auto` (default) lets `accelerate` split layers across
  CPU+GPU automatically. Use `--qwen-device cpu` to force CPU (safest,
  ~1-2 min per scenario) or `--qwen-device cuda` if you have ≥24 GB VRAM.
- `--no-keep-qwen-loaded` (default) unloads Qwen after each scenario so
  SDXL/LTX have the full VRAM to themselves. Use `--keep-qwen-loaded` if
  you generate many videos in a row and have ≥32 GB system RAM.

## Install

### 1. Custom nodes into ComfyUI

```bash
# from your ComfyUI checkout
ln -s /abs/path/to/shorts_bot/comfyui_pipeline/custom_nodes/scene_pipeline \
      ComfyUI/custom_nodes/scene_pipeline

# install the node's Python deps into ComfyUI's venv
pip install "transformers>=4.51" "accelerate>=0.33" "safetensors" "torch>=2.1"
```

`transformers>=4.51` is required for the Qwen3 chat template
(`enable_thinking=False`).

Restart ComfyUI. You should see **Qwen Scenario Generator** and
**Save Text To File** under the `scene_pipeline` category.

### 2. Orchestrator / bot / Web UI

```bash
cd shorts_bot
pip install -r comfyui_pipeline/requirements.txt
```

That pulls `requests`, `websocket-client`, `pydantic`, `gradio`, and
`python-telegram-bot`.

## Run

### CLI

```bash
export PYTHONPATH=$(pwd)
python -m comfyui_pipeline.src.cli \
    "a lonely astronaut discovers a forest on a moon" \
    --comfyui-url http://127.0.0.1:8188 \
    --output-dir output/scenes \
    --qwen-model Qwen/Qwen3-8B \
    --sdxl-checkpoint sd_xl_base_1.0.safetensors \
    --ltx-checkpoint ltx-2.3-22b-distilled-1.1.safetensors \
    --seed 42
```

Artifacts:

```
output/scenes/
├── scenario.json        # full parsed scenario
├── index.json           # manifest: scene id → image + video paths
├── scene_01.png / .mp4
├── scene_02.png / .mp4
├── concat_list.txt      # ffmpeg concat demuxer input
└── final.mp4            # all scenes stitched together
```

### Telegram bot

```bash
export TELEGRAM_BOT_TOKEN=...           # from @BotFather
export COMFYUI_URL=http://127.0.0.1:8188
python bot.py
```

Flow: `/start` → user sends a description → bot runs the pipeline → replies
with `final.mp4` in the chat (or a Google Drive link if the file exceeds
Telegram's 50 MB Bot API upload limit). Drive upload is optional; drop
`credentials.json` next to `bot.py` (or set `GOOGLE_APPLICATION_CREDENTIALS`)
to enable it.

### Gradio Web UI

```bash
python -m comfyui_pipeline.gradio_app
# open http://localhost:7860
```

Single textbox for the idea, a Generate button, and outputs for the scenario
title, per-scene descriptions, each keyframe + clip, and the final stitched
video.

## Pipeline stages in detail

### Stage 1 — Scenario (Qwen3)

`workflows/script_gen_api.json` wires:

```
PrimitiveStringMultiline (system prompt)
                                 \
PrimitiveStringMultiline (idea) ──► QwenScenarioGenerator ──► SaveTextToFile
```

`QwenScenarioGenerator` loads any Qwen-family causal LM through
`transformers.AutoModelForCausalLM`. It passes `enable_thinking=False` to the
Qwen3 chat template where supported and strips any leading `<think>…</think>`
block so downstream JSON parsing stays strict. On parse failure the
orchestrator retries once with a reminder appended to the idea.

### Stage 2 — Keyframe image (SDXL + KSamplerAdvanced)

`workflows/scene_image_api.json`:

```
CheckpointLoaderSimple
  ├─► (CLIP) CLIPTextEncode (+/−)
  └─► (MODEL, VAE) → KSamplerAdvanced → VAEDecode → SaveImage
```

`KSamplerAdvanced` is used (not the basic `KSampler`) so noise, start/end
steps and leftover-noise behaviour are explicit — useful if you later want to
split generation into multiple passes. Per scene the orchestrator patches the
positive prompt (`scene.image_prompt + ", " + scenario.style`), negative
prompt, `noise_seed`, steps, CFG, and filename prefix. The rendered PNG is
downloaded via `/view` and also re-uploaded to ComfyUI `input/` via
`/upload/image` so the next workflow can `LoadImage` it.

### Stage 3 — Scene video (LTX-2.3 Distilled, single-stage I2V)

`workflows/scene_video_api.json` implements the Lightricks single-stage
distilled I2V graph (video-only, no audio):

```
CheckpointLoaderSimple (LTX-2.3) ──► (MODEL) ──► LoraLoaderModelOnly (distilled LoRA)
                                                                      │
                                                                      ▼
CLIPLoader (Gemma 3 12B, type=ltxv) ──► (CLIP) ──► CLIPTextEncode (+/−)
                                                              │
LoadImage ──────────────────────────────► LTXVImgToVideo (width/height/length, strength)
                                                              │
                                                 (positive, negative, latent)
                                                              │
                                                        LTXVConditioning (frame_rate)
                                                              │
                                        LTXVScheduler ─► KSamplerSelect ─► SamplerCustomAdvanced
                                                         RandomNoise  ──►        │
                                                         CFGGuider   ──►         │
                                                                                 ▼
                                                                 VAEDecode ─► CreateVideo ─► SaveVideo (mp4)
```

The orchestrator computes `length = frames_for(duration_seconds)` enforcing
LTX's `(length - 1) % 8 == 0` and `length ≥ 9`. The distilled workflow uses
15 steps and `cfg=1.0` by default — the LoRA is the magic that collapses the
full 25-step chain into a fast distilled pass.

### Stage 4 — Final concatenation (FFmpeg)

After all scenes finish, `_concat_final_video()` writes a concat demuxer list
and invokes `ffmpeg -f concat -safe 0 -i concat_list.txt -c copy final.mp4`.
If stream copy fails (e.g. codecs mismatch) it automatically re-encodes with
`libx264 yuv420p crf=18`. Toggle with `PipelineConfig.concat_final_video`.

## Programmatic use

```python
from comfyui_pipeline.src.orchestrator import PipelineConfig, ScenePipeline

pipeline = ScenePipeline(PipelineConfig(
    comfyui_url="http://127.0.0.1:8188",
    output_dir="output/my_story",
    qwen_model="Qwen/Qwen3-8B",
    ltx_checkpoint="ltx-2.3-22b-distilled-1.1.safetensors",
    seed=123,
))
result = pipeline.run("a cat learns to surf during sunset")
print("final video:", result.final_video_path)
for sa in result.scene_artifacts:
    print(sa.scene.id, sa.video_path)
```

## Testing

Offline unit tests cover the schema, workflow JSON structure, orchestrator
helpers and the ffmpeg concat step (mocked) — no ComfyUI or GPU needed:

```bash
pip install pytest
pytest comfyui_pipeline/tests
```

End-to-end runs require a working ComfyUI with the models listed above, the
`scene_pipeline` custom nodes, and `ffmpeg` in `PATH`.

## Tuning

Common knobs on `PipelineConfig`:

| Field | Default | Notes |
| --- | --- | --- |
| `qwen_model` | `Qwen/Qwen3-8B` | HF ID or local path |
| `qwen_device` | `"auto"` | `"auto"` → accelerate CPU+GPU split; `"cpu"` forces CPU (low-VRAM safe); `"cuda"` pins to GPU |
| `qwen_keep_loaded` | `False` | Keep Qwen resident after scenario (disable on low-VRAM so SDXL/LTX have VRAM) |
| `sdxl_checkpoint` | `sd_xl_base_1.0.safetensors` | Any SDXL checkpoint |
| `ltx_checkpoint` | `ltx-2.3-22b-distilled-1.1.safetensors` | LTX-2.3 distilled |
| `ltx_lora` | `ltx-2.3-22b-distilled-lora-384-1.1.safetensors` | Distilled LoRA |
| `ltx_lora_strength` | `1.0` | `0.5–1.0` sensible range |
| `ltx_text_encoder` | `comfy_gemma_3_12B_it.safetensors` | Local Gemma 3 12B |
| `image_width` / `image_height` | 1024×576 | 16:9 key frame |
| `video_width` / `video_height` | 960×544 | Must be multiples of 32 |
| `video_fps` | 25.0 | Matches `LTXVConditioning.frame_rate` |
| `image_steps` / `image_cfg` | 28 / 6.5 | SDXL (KSamplerAdvanced) |
| `video_steps` / `video_cfg` | 15 / 1.0 | LTX-2.3 distilled |
| `max_script_retries` | 1 | Retries on invalid JSON from Qwen |
| `concat_final_video` | `True` | Disable to skip ffmpeg |
| `ffmpeg_binary` | `ffmpeg` | Override if ffmpeg lives elsewhere |
