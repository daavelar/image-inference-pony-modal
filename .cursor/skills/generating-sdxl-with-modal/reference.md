# IIPRS API Reference

## Overview

- **Service**: Modal (PyTorch + Diffusers SDXL pipeline)
- **Default model**: `untitledPony_b10` at `/my_vol/models/{model}.safetensors`
- **LoRAs**: `/my_vol/loras/`
- **HTTP label**: `pony-http` → `https://<user>--pony-http.modal.run`
- **Output**: Public PNG URL on Cloudflare R2 (`modal_pony_infs/` prefix)

## Endpoints

| Route | Description |
|-------|-------------|
| `GET /health` | Status, model_ready, loras_count |
| `GET /loras` | List available LoRAs |
| `POST /generate` | Async — returns 202 + call_id |
| `POST /generate?blocking=true` | Sync — blocks until complete |
| `GET /result/{call_id}` | Poll async result |

## Default Negative Prompt

Used when `negative` is omitted:

```
score_6, score_5, score_4, (worst quality, low quality:1.4), (cgpainting, cartoon, anime, 3d, render, sketch, drawing, painting, watercolor:1.2), (plastic skin:1.2), flat color, airbrushed, digital art
```

## LoRA Resolution Order

1. Exact path as provided (if file exists)
2. `/my_vol/loras/{path}`
3. Same path with `.safetensors` appended

LoRAs are unloaded after each job. The base model stays cached on warm workers.

## Recommended Ranges

| Parameter | Range | Notes |
|-----------|-------|-------|
| `steps` | 25–35 | Diminishing returns past 35 |
| `cfg` | 4.0–7.0 | Lower works well with Pony |
| `width` / `height` | 768–1536 | 1024×1024 optimal |
| LoRA `scale` | 0.5–3.0 | Sliders: 1.5–5.0; characters: 0.7–1.2 |

## Examples

### Minimal curl (sync)

```bash
curl -X POST "${MODAL_PONY_HTTP_URL}/generate?blocking=true" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "score_9, score_8_up, score_7_up, rating_safe, 1girl, smiling, portrait, natural lighting"
  }'
```

### Full curl with LoRAs

```bash
curl -X POST "${MODAL_PONY_HTTP_URL}/generate?blocking=true" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "untitledPony_b10",
    "prompt": "score_9, score_8_up, score_7_up, rating_safe, portrait, natural lighting",
    "negative": "score_6, score_5, score_4, (worst quality, low quality:1.4), cartoon, anime, 3d, render",
    "scheduler": "dpm++_sde_karras",
    "steps": 25,
    "cfg": 5.5,
    "width": 1024,
    "height": 1024,
    "seed": 42,
    "loras": [
      { "path": "example-lora.safetensors", "scale": 1.0 }
    ]
  }'
```

### Python sync

```python
import os
import requests

BASE_URL = os.environ["MODAL_PONY_HTTP_URL"]

response = requests.post(
    f"{BASE_URL}/generate",
    params={"blocking": "true"},
    headers={"Content-Type": "application/json"},
    json={
        "prompt": "score_9, score_8_up, score_7_up, rating_safe, 1girl, portrait",
        "steps": 30,
        "cfg": 5.0,
        "seed": 12345,
    },
    timeout=300,
)

data = response.json()
if data.get("error"):
    raise RuntimeError(data["error"])

print(f"URL: {data['url']}")
print(f"Seed: {data['seed']}")
print(f"Cost: ${data['cost']:.6f}")
```

### Python async with polling

```python
import os
import time
import requests

BASE_URL = os.environ["MODAL_PONY_HTTP_URL"]

job = requests.post(
    f"{BASE_URL}/generate",
    headers={"Content-Type": "application/json"},
    json={"prompt": "score_9, score_8_up, 1girl, smiling"},
).json()

call_id = job["call_id"]

while True:
    status = requests.get(f"{BASE_URL}/result/{call_id}").json()
    if status.get("status") == "completed":
        print(f"URL: {status['url']}")
        break
    if status.get("status") == "failed" or status.get("error"):
        raise RuntimeError(status.get("error", "Job failed"))
    time.sleep(2)
```

### TypeScript sync

```typescript
const BASE_URL = process.env.MODAL_PONY_HTTP_URL!;

const response = await fetch(
  `${BASE_URL}/generate?blocking=true`,
  {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      prompt: "score_9, score_8_up, score_7_up, rating_safe, 1girl, portrait",
      steps: 30,
      cfg: 5.0,
    }),
  }
);

const data = await response.json();
console.log(data.url, data.seed);
```

## Common Errors

| Error | Cause |
|-------|-------|
| `prompt (non-empty string) required` | Missing or empty `prompt` |
| `Model not found: ...` | `{model}.safetensors` missing from volume |
| `Lora not found: ...` | LoRA file missing from volume |
| `S3 upload failed: ...` | R2 credentials missing or invalid |

## Local Testing

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8003
curl -X POST http://localhost:8003/generate -H "Content-Type: application/json" -d @input.json
```

## Modal CLI Testing

```bash
export MODAL_AWS_SECRET=pony-r2
modal run generate-modal.py --payload-json '{"prompt":"score_9, portrait, natural lighting"}'
```

## Deploy

```bash
export MODAL_AWS_SECRET=pony-r2
modal deploy generate-modal.py
```
