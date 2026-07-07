#!/usr/bin/env python3
"""
Script para baixar o modelo Untitled Pony v8 do Hugging Face
"""

import os
import subprocess
from pathlib import Path


def download_model():
    """Baixa o modelo Untitled Pony v8 usando wget ou curl"""

    # Configurações
    url = "https://huggingface.co/aimatch/untitled-pony-v8/resolve/main/untitledPony_b10.safetensors"
    models_dir = Path(__file__).parent / "models"
    output_file = models_dir / "untitledPony_b10.safetensors"

    # Criar diretório models se não existir
    models_dir.mkdir(exist_ok=True)

    print(f"Baixando modelo para {output_file}...")

    # Tentar com wget primeiro, depois curl
    try:
        subprocess.run(
            ["wget", "-q", "--show-progress", "-O", str(output_file), url],
            check=True,
        )
    except FileNotFoundError:
        try:
            subprocess.run(
                ["curl", "-L", "-o", str(output_file), url],
                check=True,
            )
        except FileNotFoundError:
            print("Erro: wget ou curl não encontrados. Instale um deles.")
            return False

    print(f"✓ Arquivo baixado com sucesso em: {output_file}")
    return True


if __name__ == "__main__":
    download_model()
