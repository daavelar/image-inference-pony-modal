# Image Inference Pony — Modal (IIPRS)

Endpoint [Modal](https://modal.com) para geração de imagens com modelos SDXL/Pony Diffusion. O worker carrega o checkpoint em GPU, aplica LoRAs opcionais, faz upload da imagem para Cloudflare R2 e retorna a URL pública.

## Como funciona

```
Cliente → POST /generate (Modal HTTP)
       → PonyWorker carrega modelo + LoRAs (Diffusers)
       → inferência SDXL
       → upload R2
       → resposta { url, key, seed, duration_seconds, cost }
```

O modelo base fica em cache na GPU entre requisições. LoRAs são carregadas por job e descarregadas ao final.

## Estrutura do projeto

```
.
├── generate-modal.py       # Entrypoint Modal (worker GPU + HTTP)
├── app/
│   ├── config.py           # Settings (R2, paths, defaults Pony)
│   ├── pony_pipeline.py    # Diffusers SDXL pipeline
│   ├── inference.py        # Handler de geração + upload
│   ├── ip_face.py          # IP-Adapter, style LoRAs e face detailer
│   ├── storage.py          # Upload boto3 → R2
│   ├── schemas.py          # Contratos request/response
│   └── main.py             # FastAPI local (dev)
├── scripts/
│   ├── upload-models-to-modal-volume.py
│   └── upload-loras-to-modal-volume.py
├── input.json              # Exemplo de payload flat
└── .env.example            # Chaves R2 (copiar para .env)
```

## Pré-requisitos

1. Conta Modal: `pip install modal && modal setup`
2. Volume Modal `ponyinf` com modelos e LoRAs
3. Secret Modal `pony-r2` com credenciais R2

### Layout no volume Modal

| Caminho no volume | Conteúdo |
|-------------------|----------|
| `/my_vol/models/{nome}.safetensors` | Checkpoint SDXL/Pony |
| `/my_vol/loras/` | Arquivos LoRA (`.safetensors`) |
| `/my_vol/ipadapter/image_encoder/` | CLIP ViT-H (`config.json`, `model.safetensors`) |
| `/my_vol/ipadapter/sdxl_models/` | `ip-adapter-plus-face_sdxl_vit-h.safetensors`, `ip-adapter-plus_sdxl_vit-h.safetensors` |
| `/my_vol/detectors/face_yolov8m.pt` | Detector de rosto do face detailer |

Modelo padrão: `untitledPony_b10` → `/my_vol/models/untitledPony_b10.safetensors`

### Upload de modelos e LoRAs

```bash
python scripts/upload-models-to-modal-volume.py   # ~7 GB, demorado
python scripts/upload-loras-to-modal-volume.py
modal volume ls ponyinf /models
modal volume ls ponyinf /loras
```

### Secret R2 na Modal

```bash
modal secret create pony-r2 \
  AWS_BUCKET=... \
  AWS_DEFAULT_REGION=auto \
  AWS_ENDPOINT=... \
  AWS_URL=... \
  AWS_ACCESS_KEY_ID=... \
  AWS_SECRET_ACCESS_KEY=... \
  AWS_USE_PATH_STYLE_ENDPOINT=false
```

## Deploy

```bash
export MODAL_AWS_SECRET=pony-r2
modal deploy generate-modal.py
```

URL HTTP: `https://<seu-usuario>--pony-http.modal.run`

## API

### Endpoints

| Rota | Descrição |
|------|-----------|
| `GET /health` | Status + model_ready + loras_count |
| `GET /loras` | Lista LoRAs no volume |
| `POST /generate` | Async (202 + call_id) |
| `POST /generate?blocking=true` | Sync — aguarda resultado |
| `GET /result/{call_id}` | Poll do resultado async |

### Request

Payload **flat** (sem wrapper `input`):

```json
{
  "prompt": "score_9, score_8_up, score_7_up, rating_safe, 1girl, portrait",
  "model": "untitledPony_b10",
  "negative": "score_6, score_5, score_4, (worst quality, low quality:1.4)",
  "scheduler": "dpm++_sde_karras",
  "steps": 25,
  "cfg": 5.5,
  "width": 1024,
  "height": 1024,
  "seed": -1,
  "loras": [
    { "path": "example-lora.safetensors", "scale": 1.0 }
  ],
  "style_loras": true,
  "ip_adapter": { "image": "<base64>", "weight": 0.4, "model": "plus_face" },
  "face_detailer": {
    "faces": [{ "image": "<base64>", "weight": 0.4 }],
    "denoise": 0.4
  },
  "callback_url": "https://your-app.example/api/webhooks/modal/image",
  "external_id": 123
}
```

Campos opcionais de webhook:
- `callback_url` — URL que recebe POST ao concluir (alternativa ao polling)
- `external_id` — ID repassado no payload do webhook

Veja `input.json` para um exemplo completo.

### Response (sucesso)

```json
{
  "url": "https://.../modal_pony_infs/20260707/uuid.png",
  "key": "modal_pony_infs/20260707/uuid.png",
  "seed": 123456789,
  "duration_seconds": 12.5,
  "cost": 0.002775
}
```

### Response (erro)

```json
{ "error": "prompt (non-empty string) required" }
```

## Parâmetros

| Campo | Obrigatório | Padrão | Descrição |
|-------|-------------|--------|-----------|
| `prompt` | sim | — | Descrição da imagem |
| `model` | não | `untitledPony_b10` | Nome do arquivo sem `.safetensors` |
| `negative` | não | negative Pony embutido | Prompt negativo. Alias: `negative_prompt` |
| `scheduler` | não | `euler_a` | Scheduler de amostragem |
| `steps` | não | `30` | Passos de inferência (25–35 recomendado) |
| `cfg` | não | `5.0` | Guidance scale (4.0–7.0) |
| `width` / `height` | não | `1024` | Dimensões em pixels |
| `seed` | não | aleatório | `-1` ou omitido = aleatório; retorna seed real usado |
| `loras` | não | `[]` | Lista de `{ "path": "arquivo.safetensors", "scale": 1.0 }` |
| `style_loras` | não | — | `true` aplica os presets configurados em 1.0. Objeto escolhe a chave (`preset_a`, `preset_b`, `preset_c`) e o peso (`null` = 1.0). O arquivo de cada um vem de `STYLE_LORA_A`, `STYLE_LORA_B` e `STYLE_LORA_C` |
| `ip_adapter` | não | — | `{ "image": "<base64>", "weight": 0.4, "model": "plus_face" }` ou `"plus"` |
| `face_detailer` | não | — | Repinta rostos da esquerda para a direita. `faces[]` com `image` e `weight`. Opcionais: `denoise` (0.4), `steps` (20), `padding` (1.8), `confidence` (0.3). `base_image` pula o txt2img e só repinta as máscaras |

### Schedulers disponíveis

`euler_a`, `dpm++_sde_karras`, `dpm++_2m_karras`, `heun`, `ddim`, `lms`

## Teste CLI

```bash
modal run generate-modal.py --payload-json '{"prompt":"score_9, portrait, natural lighting"}'
```

### Exemplo curl (sync)

```bash
curl -X POST "https://<user>--pony-http.modal.run/generate?blocking=true" \
  -H "Content-Type: application/json" \
  -d @input.json
```

### Exemplo curl (async)

```bash
# Submit
curl -X POST "https://<user>--pony-http.modal.run/generate" \
  -H "Content-Type: application/json" \
  -d @input.json

# Poll
curl "https://<user>--pony-http.modal.run/result/{call_id}"
```

## Desenvolvimento local

Copie `.env.example` para `.env` com credenciais R2. Coloque modelos em `models/` e LoRAs em `loras/`.

```bash
pip install fastapi uvicorn pydantic-settings python-dotenv boto3 \
  diffusers transformers accelerate safetensors peft Pillow torch

uvicorn app.main:app --host 0.0.0.0 --port 8003
```

## Variáveis de ambiente

| Variável | Default | Uso |
|----------|---------|-----|
| `MODAL_VOLUME_NAME` | `ponyinf` | Nome do volume Modal |
| `MODAL_VOLUME_MOUNT` | `/my_vol` | Ponto de montagem |
| `MODAL_AWS_SECRET` | — | Nome do secret R2 na Modal |
| `MODAL_GPU` | `A100,L40S,A10,L4` | GPU do worker (fallback em ordem, mais forte primeiro) |
| `COST_PER_SECOND` | conforme GPU primária | Custo estimado por segundo (override manual) |

### GPUs disponíveis (ordem padrão de fallback)

| GPU | Custo/s |
|-----|---------|
| Nvidia A100 40 GB | $0.000583 |
| Nvidia L40S | $0.000542 |
| Nvidia A10 | $0.000306 |
| Nvidia L4 | $0.000222 |

## Skill para agentes

```
.cursor/skills/generating-sdxl-with-modal/
├── SKILL.md
└── reference.md
```
