import os
import torch
import warnings
import base64
from io import BytesIO
from pathlib import Path
import runpod
import requests
from tqdm import tqdm
import sys

# Monkeypatch para evitar erro 'AttributeError: module torch has no attribute xpu'
# em algumas versões do diffusers/torch
if not hasattr(torch, "xpu"):
    class XPUStub:
        def is_available(self): return False
        def empty_cache(self): pass
    torch.xpu = XPUStub()
    sys.modules["torch.xpu"] = torch.xpu

from diffusers import (
    StableDiffusionXLPipeline, 
    EulerAncestralDiscreteScheduler, 
    DPMSolverMultistepScheduler,
    DDIMScheduler,
    LMSDiscreteScheduler,
    HeunDiscreteScheduler
)

# Silencia avisos do PEFT
warnings.filterwarnings("ignore", message=".*Already found a `peft_config` attribute.*")

# --- Globais para Cache do Modelo ---
pipe = None
current_model = None

def get_scheduler(pipe_obj, scheduler_name):
    config = pipe_obj.scheduler.config
    if scheduler_name == "euler_a":
        return EulerAncestralDiscreteScheduler.from_config(config)
    elif scheduler_name == "dpm++_sde_karras":
        return DPMSolverMultistepScheduler.from_config(
            config, algorithm_type="sde-dpmsolver++", use_karras_sigmas=True
        )
    elif scheduler_name == "dpm++_2m_karras":
        return DPMSolverMultistepScheduler.from_config(config, use_karras_sigmas=True)
    elif scheduler_name == "heun":
        return HeunDiscreteScheduler.from_config(config)
    elif scheduler_name == "ddim":
        return DDIMScheduler.from_config(config)
    elif scheduler_name == "lms":
        return LMSDiscreteScheduler.from_config(config)
    return EulerAncestralDiscreteScheduler.from_config(config)


def load_model(model_name="lucentxl_pony_by_klaabu"):
    """Carrega o modelo globalmente para persistência na instância serverless."""
    global pipe, current_model

    # Se o modelo em cache é o mesmo, retorna
    if pipe is not None and current_model == model_name:
        return pipe

    model_path = f"/runpod-volume/models/{model_name}.safetensors"

    # O modelo deve estar presente no volume montado (/runpod-volume)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo não encontrado em {model_path}. Certifique-se de que o volume está montado corretamente.")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Iniciando carregamento do modelo: {model_path}")

    pipe = StableDiffusionXLPipeline.from_single_file(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        use_safetensors=True
    )

    # Otimizações padrão para Serverless (RTX 4090 ou similar do RunPod)
    pipe.to(device)

    # Se precisar de xformers (opcional)
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    current_model = model_name
    print("Modelo carregado com sucesso!")
    return pipe

def handler(job):
    """Handler principal do RunPod Serverless."""
    global pipe
    job_input = job["input"]

    # Parâmetros com valores padrão
    prompt = job_input.get("prompt")
    model = job_input.get("model", "lucentxl_pony_by_klaabu")
    negative_prompt = job_input.get("negative") or job_input.get("negative_prompt", "score_6, score_5, score_4, (worst quality, low quality:1.4), (cgpainting, cartoon, anime, 3d, render, sketch, drawing, painting, watercolor:1.2), (plastic skin:1.2), flat color, airbrushed, digital art")
    scheduler_name = job_input.get("scheduler", "euler_a")
    steps = job_input.get("steps", 30)
    cfg = job_input.get("cfg", 5.0)
    width = job_input.get("width", 1024)
    height = job_input.get("height", 1024)
    seed = job_input.get("seed", -1)

    loras = job_input.get("loras", []) # Lista de objetos {"path": "...", "scale": 1.0}

    if not prompt:
        return {"error": "O campo 'prompt' é obrigatório."}

    # 1. Garante que o modelo está carregado
    pipe = load_model(model)

    # 2. Configura Scheduler
    pipe.scheduler = get_scheduler(pipe, scheduler_name)

    # 3. Gerencia LoRAs
    if loras:
        adapter_names = []
        adapter_weights = []
        
        # Limpa adaptadores anteriores se necessário (opcional dependendo do uso)
        # pipe.unload_lora_weights() 
        
        for i, lora_item in enumerate(loras):
            lora_path = lora_item["path"]
            lora_scale = lora_item.get("scale", 1.0)
            
            # Busca inteligente (Prioridade: Volume rede -> Local)
            if not os.path.exists(lora_path):
                # Define os caminhos de busca em ordem de prioridade
                search_paths = [
                    os.path.join("/runpod-volume/loras", lora_path),
                    os.path.join("loras", lora_path)
                ]
                
                found = False
                for path in search_paths:
                    if os.path.exists(path):
                        lora_path = path
                        found = True
                        break
                
                if not found and not lora_path.endswith(".safetensors"):
                    # Tenta com extensão .safetensors se não encontrou nada
                    for base in ["/runpod-volume/loras", "loras", ""]:
                        test_path = os.path.join(base, lora_path + ".safetensors") if base else lora_path + ".safetensors"
                        if os.path.exists(test_path):
                            lora_path = test_path
                            break

            adapter_name = f"adapter_{i}"
            try:
                pipe.load_lora_weights(lora_path, adapter_name=adapter_name)
                adapter_names.append(adapter_name)
                adapter_weights.append(lora_scale)
            except Exception as e:
                print(f"Erro LoRA {lora_path}: {e}")

        if adapter_names:
            pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)

    # 4. Inferência
    generator = torch.Generator(device="cuda").manual_seed(seed) if seed != -1 else None
    
    with torch.inference_mode():
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=cfg,
            width=width,
            height=height,
            generator=generator
        ).images[0]

    # 5. Retorno (Base64 para Serverless)
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Limpa adaptadores para o próximo job não acumular
    if loras:
        pipe.unload_lora_weights()

    return {
        "image": img_str,
        "seed": seed
    }

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
