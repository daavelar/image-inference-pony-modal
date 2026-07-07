#!/usr/bin/env python3
"""
Upload local Pony SDXL checkpoint(s) into a Modal Volume for generate-modal.py.

Expected layout inside the volume:
    /models/<model>.safetensors

Environment (optional):
    MODAL_VOLUME_NAME   — default ponyinf
    MODAL_MODELS_LOCAL  — default <repo>/models
    MODAL_MODELS_REMOTE — path inside the volume root, default /models

CLI:
    pip install modal
    modal setup  # once
    python scripts/upload-models-to-modal-volume.py
    python scripts/upload-models-to-modal-volume.py --local ./models --remote /models

Or use Modal CLI:
    modal volume put ponyinf ./models/untitledPony_b10.safetensors /models/
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_SUFFIXES = (".safetensors",)


def _collect_model_files(local_dir: Path, paths: list[Path]) -> list[Path]:
    if paths:
        files: list[Path] = []
        for raw in paths:
            candidate = raw.expanduser()
            if not candidate.is_absolute():
                candidate = (Path.cwd() / candidate).resolve()
            else:
                candidate = candidate.resolve()

            if candidate.is_dir():
                files.extend(
                    sorted(
                        path
                        for path in candidate.iterdir()
                        if path.is_file() and path.suffix.lower() in MODEL_SUFFIXES
                    )
                )
                continue

            if not candidate.is_file():
                raise FileNotFoundError(f"Model file not found: {candidate}")
            if candidate.suffix.lower() not in MODEL_SUFFIXES:
                raise ValueError(
                    f"Unsupported model extension (expected .safetensors): {candidate.name}"
                )
            files.append(candidate)
        return files

    if not local_dir.is_dir():
        raise FileNotFoundError(f"Models directory not found: {local_dir}")

    files = sorted(
        path
        for path in local_dir.iterdir()
        if path.is_file() and path.suffix.lower() in MODEL_SUFFIXES
    )
    if not files:
        raise FileNotFoundError(f"No .safetensors files found in {local_dir}")
    return files


def main(argv: list[str] | None = None) -> int:
    default_vol = os.environ.get("MODAL_VOLUME_NAME", "ponyinf")
    default_local = Path(os.environ.get("MODAL_MODELS_LOCAL", REPO_ROOT / "models"))
    default_remote = os.environ.get("MODAL_MODELS_REMOTE", "/models").strip() or "/"

    p = argparse.ArgumentParser(description="Upload Pony SDXL checkpoints to a Modal Volume.")
    p.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="Model file(s) or directory to upload (default: all .safetensors in ./models)",
    )
    p.add_argument("--volume", "-v", default=default_vol, help="Modal volume name")
    p.add_argument(
        "--local",
        "-l",
        type=Path,
        default=default_local,
        help="Local models directory when no files are passed",
    )
    p.add_argument(
        "--remote",
        "-r",
        default=default_remote,
        help="Destination path inside the volume (e.g. /models)",
    )
    p.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Overwrite existing files in the volume",
    )
    args = p.parse_args(argv)

    local_dir = args.local.expanduser().resolve()
    remote_root = args.remote.strip() or "/"
    remote_root = "/" if remote_root in ("", "/") else "/" + remote_root.strip("/")

    try:
        model_files = _collect_model_files(local_dir, args.files)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        import modal
    except ImportError:
        print("error: install the Modal client: pip install modal", file=sys.stderr)
        return 1

    vol = modal.Volume.from_name(args.volume, create_if_missing=True)
    print(
        f"Uploading {len(model_files)} model file(s) to volume {args.volume!r} at {remote_root!r}:",
        flush=True,
    )
    for path in model_files:
        size_gb = path.stat().st_size / (1024**3)
        print(f"  {path.name} ({size_gb:.2f} GB)", flush=True)
    print("(large uploads can take a long time)", flush=True)

    with vol.batch_upload(force=args.force) as batch:
        for path in model_files:
            batch.put_file(str(path), f"{remote_root}/{path.name}")

    print(f"Done. Example: modal volume ls {args.volume} {remote_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
