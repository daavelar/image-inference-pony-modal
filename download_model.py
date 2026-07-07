import requests
import os
from tqdm import tqdm
from pathlib import Path

def download_model(url, output_path):
    """
    Baixa o modelo do Hugging Face com barra de progresso.
    """
    # Garante que o diretório de destino existe
    target_dir = Path(output_path).parent
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Conectando ao Hugging Face...")
    try:
        # Resolve a URL do Hugging Face para o link de download direto se necessário
        # Mas a URL fornecida já aponta para o blob, o requests segue redirecionamentos
        download_url = url.replace("/blob/", "/resolve/")
        
        response = requests.get(download_url, stream=True, allow_redirects=True)
        response.raise_for_status()
    except Exception as e:
        print(f"Erro ao conectar: {e}")
        return

    filename = url.split('/')[-1]
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 # 1 Kibibyte

    print(f"Baixando: {filename}")
    print(f"Para: {output_path}")

    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True, desc=filename)
    
    with open(output_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    
    progress_bar.close()

    if total_size != 0 and progress_bar.n != total_size:
        print("ERRO: O download falhou ou foi interrompido.")
    else:
        print(f"Download concluído com sucesso!")

if __name__ == "__main__":
    MODEL_URL = "https://huggingface.co/klaabu/lucent-xl/resolve/main/lucentxl_pony_by_klaabu.safetensors"
    OUTPUT_PATH = "models/lucentxl_pony_by_klaabu.safetensors"

    download_model(MODEL_URL, OUTPUT_PATH)
