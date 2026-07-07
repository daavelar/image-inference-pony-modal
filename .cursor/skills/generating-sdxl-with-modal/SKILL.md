---
name: generating-image-with-iiprs
description: Generates images via the IIPRS Modal API (Pony SDXL). Use when the user asks to generate, create, or render images with IIPRS, image-inference-pony-modal, Pony Diffusion, or the Modal Pony image inference endpoint.
---

# Generating Image with IIPRS

IIPRS (Image Inference Pony Modal) is a Modal HTTP endpoint that generates PNG images using SDXL/Pony Diffusion models. Responses return a public R2 URL instead of base64.

## Prerequisites

Require this environment variable before calling the API:

| Variable | Description |
|----------|-------------|
| `MODAL_PONY_HTTP_URL` | Base URL of the deployed Modal HTTP app (e.g. `https://<user>--pony-http.modal.run`) |

If missing, ask the user to provide it before proceeding.

## Workflow

1. **Build the prompt** using Pony conventions (see below).
2. **Choose parameters** — use defaults unless the user specifies otherwise.
3. **Call the API** — prefer `?blocking=true` for single images; use async + poll for batches.
4. **Use the response URL** — `url` points to the PNG on R2; `key` is the object key.
5. **Report the seed** — return `seed` so the user can reproduce the result.

## API Call (sync — default)

```bash
curl -X POST "${MODAL_PONY_HTTP_URL}/generate?blocking=true" \
  -H "Content-Type: application/json" \
  -d @payload.json
```

`payload.json` structure (flat JSON, no `input` wrapper):

```json
{
  "prompt": "score_9, score_8_up, score_7_up, rating_safe, 1girl, portrait, natural lighting",
  "negative": "score_6, score_5, score_4, (worst quality, low quality:1.4), cartoon, anime, 3d, render",
  "model": "untitledPony_b10",
  "scheduler": "dpm++_sde_karras",
  "steps": 25,
  "cfg": 5.5,
  "width": 1024,
  "height": 1024,
  "seed": -1,
  "loras": [
    { "path": "pony-realism-slider.safetensors", "scale": 2.5 }
  ]
}
```

## Parameters

| Field | Required | Default | Notes |
|-------|----------|---------|-------|
| `prompt` | **yes** | — | Image description |
| `model` | no | `untitledPony_b10` | Filename without `.safetensors` |
| `negative` | no | built-in Pony negative | Alias: `negative_prompt` |
| `scheduler` | no | `euler_a` | `dpm++_sde_karras` recommended for realism |
| `steps` | no | `30` | Use 25–35 |
| `cfg` | no | `5.0` | Use 4.0–7.0 for Pony models |
| `width` / `height` | no | `1024` | SDXL sweet spot is 1024×1024 |
| `seed` | no | random | `-1` or omit = random; response includes actual seed used |
| `loras` | no | `[]` | Array of `{ "path": "file.safetensors", "scale": 1.0 }` |

Schedulers: `euler_a`, `dpm++_sde_karras`, `dpm++_2m_karras`, `heun`, `ddim`, `lms`.

## Pony Prompt Conventions

Always prepend quality tags:

```
score_9, score_8_up, score_7_up
```

Add a rating tag when appropriate: `rating_safe`, `rating_safe`, `rating_safe`.

Include quality degradation tags in the negative prompt:

```
score_6, score_5, score_4
```

**Template:**

```
score_9, score_8_up, score_7_up, rating_[level], photo of [subject], [details], natural lighting, shot on 35mm lens
```

## LoRAs

LoRA files resolve from the Modal volume at `/my_vol/loras/`. Pass only the filename unless a full path is needed.

| LoRA type | Typical `scale` |
|-----------|-----------------|
| Character LoRA | 0.7–1.2 |
| Style/slider LoRA | 1.5–5.0 |

Stack multiple LoRAs by adding entries to the `loras` array. See `input.json` in the repo root for a real example.

## Response Handling

**Success:**

```json
{
  "url": "https://.../modal_pony_infs/20260707/uuid.png",
  "key": "modal_pony_infs/20260707/uuid.png",
  "seed": 42,
  "duration_seconds": 12.5,
  "cost": 0.002775
}
```

**Error:**

```json
{ "error": "prompt (non-empty string) required" }
```

Download the image from `url` directly — no base64 decoding needed.

## Async (batch jobs)

1. `POST /generate` — submit job, get `call_id`
2. `GET /result/{call_id}` — poll until `status: completed` or `failed`

## Decision Guide

| Scenario | Mode |
|----------|------|
| Single image, user waiting | `POST /generate?blocking=true` |
| Multiple images in parallel | Async + poll |
| Reproducible result | Set explicit `seed` |
| Realistic photo style | `dpm++_sde_karras`, cfg 5.0–5.5, realism LoRAs |

## Additional Resources

- Full API reference, curl/Python/TS examples: [reference.md](reference.md)
- Example payload: `input.json`
- Entrypoint source: `generate-modal.py`, `app/inference.py`
