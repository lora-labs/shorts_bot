# ComfyUI Scene Pipeline

End-to-end pipeline that turns a **text idea** into a set of short scene videos
via ComfyUI. Stages:

```
text idea
  └─► Qwen (local)  ──► JSON scenario ──► for each scene:
                                           ├─► SDXL (keyframe image)
                                           └─► LTX-Video (img → video)
```

The orchestrator is a plain Python script that talks to a running ComfyUI via
its HTTP + WebSocket API. Two custom nodes (`QwenScenarioGenerator`,
`SaveTextToFile`) are installed into ComfyUI so the LLM stage runs fully
locally.

## Layout

```
comfyui_pipeline/
├── custom_nodes/scene_pipeline/    # drop-in ComfyUI custom node package
│   ├── __init__.py
│   ├── qwen_node.py                #   QwenScenarioGenerator
│   └── text_save_node.py           #   SaveTextToFile
├── workflows/                      # ComfyUI "API format" workflows
│   ├── script_gen_api.json         #   idea → Qwen → JSON
│   ├── scene_image_api.json        #   prompt → SDXL → PNG
│   └── scene_video_api.json        #   image + prompt → LTX → MP4
├── prompts/system_prompt.txt       # JSON-only system prompt for Qwen
├── src/
│   ├── comfy_client.py             # minimal HTTP + WS client
│   ├── orchestrator.py             # pipeline glue
│   ├── schema.py                   # Pydantic Scenario/Scene models
│   └── cli.py                      # `python -m comfyui_pipeline.src.cli ...`
├── tests/                          # offline unit tests (no ComfyUI required)
└── requirements.txt
```

## Prerequisites

* A running ComfyUI (>= 0.3, the one bundled with core LTX support) reachable
  on `http://127.0.0.1:8188` (or wherever — configurable).
* Python 3.10+ on the machine running the orchestrator (can be the same box).
* CUDA GPU recommended. Rough VRAM floor for the defaults:
  * SDXL base: ~8 GB
  * LTX-Video 2B 0.9: ~10–12 GB
  * Qwen2.5-7B-Instruct bf16: ~15 GB (use a smaller Qwen or a quant if tight).
  * The pipeline runs image + video workflows *sequentially*, so the peak is
    whichever stage needs more, not the sum.

### Required model files (place in your ComfyUI install)

| Path under `ComfyUI/models/` | Source |
| --- | --- |
| `checkpoints/sd_xl_base_1.0.safetensors` | <https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0> |
| `checkpoints/ltx-video-2b-v0.9.safetensors` | <https://huggingface.co/Lightricks/LTX-Video> |

The Qwen model is loaded through `transformers`; by default
`Qwen/Qwen2.5-7B-Instruct` is pulled from the Hugging Face hub on first run.
Swap via `--qwen-model` / `PipelineConfig.qwen_model` (e.g. a local path or
`Qwen/Qwen2.5-3B-Instruct` if you want something lighter).

## Install

### 1. Install the custom nodes into ComfyUI

```bash
# from your ComfyUI checkout
ln -s /abs/path/to/shorts_bot/comfyui_pipeline/custom_nodes/scene_pipeline \
      ComfyUI/custom_nodes/scene_pipeline
# or just copy the directory

# install the node's Python deps into ComfyUI's venv
pip install "transformers>=4.45" "accelerate>=0.33" "safetensors" "torch>=2.1"
```

Restart ComfyUI. You should now see **Qwen Scenario Generator** and
**Save Text To File** under the `scene_pipeline` category.

### 2. Install the orchestrator

```bash
cd shorts_bot
pip install -r comfyui_pipeline/requirements.txt
```

## Run

```bash
export PYTHONPATH=$(pwd)          # so `comfyui_pipeline` is importable
python -m comfyui_pipeline.src.cli \
    "a lonely astronaut discovers a forest on a moon" \
    --comfyui-url http://127.0.0.1:8188 \
    --output-dir output/scenes \
    --qwen-model Qwen/Qwen2.5-7B-Instruct \
    --sdxl-checkpoint sd_xl_base_1.0.safetensors \
    --ltx-checkpoint ltx-video-2b-v0.9.safetensors \
    --seed 42
```

The script prints one line per rendered scene. Artifacts land in
`output/scenes/`:

```
output/scenes/
├── scenario.json        # full parsed scenario
├── index.json           # manifest: scene id → image + video paths
├── scene_01.png         # SDXL keyframe
├── scene_01.mp4         # LTX video
├── scene_02.png
├── scene_02.mp4
└── ...
```

## How it fits together

### Stage 1 — Scenario (Qwen)

`workflows/script_gen_api.json` wires:

```
PrimitiveStringMultiline (system prompt)
                                 \
PrimitiveStringMultiline (idea) ──► QwenScenarioGenerator ──► SaveTextToFile
```

The orchestrator calls `POST /prompt`, waits on the websocket until the prompt
finishes, then reads `outputs[<qwen_node>].text[0]` from `/history/{id}` and
parses it as a `Scenario` (see `schema.py`). On a parse failure it retries once
with a stricter reminder appended to the user prompt.

### Stage 2 — Keyframe image (SDXL)

`workflows/scene_image_api.json` is a standard SDXL txt2img graph:

```
CheckpointLoaderSimple
  ├─► (CLIP) CLIPTextEncode (+/−)
  └─► (MODEL, VAE) → KSampler → VAEDecode → SaveImage
```

For each scene the orchestrator patches the positive prompt
(`scene.image_prompt + ", " + scenario.style`), the negative prompt, the
KSampler seed, and the SaveImage filename prefix. The rendered PNG is
downloaded via `/view` and also uploaded back into ComfyUI's `input/` dir via
`/upload/image` so the next workflow can `LoadImage` it.

### Stage 3 — Scene video (LTX-Video)

`workflows/scene_video_api.json` follows the core ComfyUI LTX example:

```
CheckpointLoaderSimple (LTX)
   ├─► (MODEL) ──────────────────────────────────────► SamplerCustom
   ├─► (CLIP) → CLIPTextEncode (+/−) → LTXVConditioning ─┐
   └─► (VAE) ──────┐                                      │
LoadImage  ───────►│                                      │
                   └─► LTXVImgToVideo ─► LTXVScheduler ──►│
                        │                                 │
                        └──► SamplerCustom ─► VAEDecode ─► CreateVideo ─► SaveVideo (mp4)
```

The orchestrator computes `length = frames_for(duration_seconds)` enforcing
`(length - 1) % 8 == 0` and `length >= 9` (LTX requirement). Output mp4 is
downloaded to `output/scenes/scene_{NN}.mp4`.

## Programmatic use

```python
from comfyui_pipeline.src.orchestrator import PipelineConfig, ScenePipeline

pipeline = ScenePipeline(PipelineConfig(
    comfyui_url="http://127.0.0.1:8188",
    output_dir="output/my_story",
    qwen_model="Qwen/Qwen2.5-7B-Instruct",
    seed=123,
))
result = pipeline.run("a cat learns to surf during sunset")
for sa in result.scene_artifacts:
    print(sa.scene.id, sa.video_path)
```

Hooking the pipeline up to the existing `shorts_bot` Telegram handler is a
drop-in replacement for the stubbed `generate_video()` function.

## Testing

Offline unit tests cover the schema, workflow JSON structure, and orchestrator
helpers — no ComfyUI needed:

```bash
pip install pytest
pytest comfyui_pipeline/tests
```

End-to-end runs require a working ComfyUI with the models above and the
`scene_pipeline` custom nodes installed.

## Tuning

Common knobs on `PipelineConfig`:

| Field | Default | Notes |
| --- | --- | --- |
| `image_width` / `image_height` | 1024×576 | 16:9 key frame |
| `video_width` / `video_height` | 768×512 | Must be multiples of 32 |
| `video_length_frames` | 97 | Overridden per-scene from `duration_seconds` |
| `video_fps` | 25.0 | Matches `LTXVConditioning.frame_rate` |
| `image_steps` / `image_cfg` | 28 / 6.5 | SDXL sampler |
| `video_steps` / `video_cfg` | 30 / 3.0 | LTX sampler |
| `max_script_retries` | 1 | Retries on invalid JSON from Qwen |

If you need a lighter LTX model (e.g. on 12 GB), you can use the distilled
variants from the Lightricks repo — just swap `--ltx-checkpoint`.
