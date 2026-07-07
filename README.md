# SDXL Image Inference

Este diretório contém o script para gerar imagens usando modelos SDXL (como o Pony Diffusion) no formato `.safetensors`.

## Como usar

1.  **Instale as dependências:**

    ```bash
    pip install -r requirements.txt
    # Caso o erro de vulnerabilidade persista (CVE-2025-32434):
    pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```


    ```bash
    ```

3.  **Rode o script de geração:**
    ```bash
    python generate_sdxl.py \
      --model "/caminho/do/modelo.safetensors" \
      --prompt "sua descrição aqui" \
      --output "outputs/resultado.png"
    ```

## Parâmetros Principais

- `--model`: (Opcional) Caminho para o arquivo `.safetensors`. Padrão: `models/untitled-pony-v10.safetensors`.
- `--prompt`: (Obrigatório) O que você quer gerar.
- `--lora`: (Opcional) Caminho para um arquivo LoRA.
- `--lora_scale`: (Opcional) Peso da LoRA (padrão 1.0).
- `--steps`: Número de passos de inferência (recomendado 28-35).
- `--cfg`: Guidance scale (recomendado 7.0).
