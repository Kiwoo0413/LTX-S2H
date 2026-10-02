"""
scripts/download_models.py
Utility script to check and download required LTX-2.5 model components.
"""

import argparse
import sys
from pathlib import Path

# Add library root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.model_manager import (
    DEFAULT_LORA_FILENAME,
    DEFAULT_SCENE_EMB_FILENAME,
    DEFAULT_TRANSFORMER_FILENAME,
    DEFAULT_VAE_FILENAME,
    HF_BASE_REPO,
    HF_IC_LORA_REPO,
    LTXModelManager,
)


def print_status(custom_dir: str = None) -> None:
    paths = LTXModelManager.resolve_all_paths(custom_model_dir=custom_dir)
    print("=" * 60)
    print(" LTX-2.5 SDR-to-HDR Model Status Check")
    if custom_dir:
        print(f" Target Directory      : {custom_dir}")
    print("=" * 60)
    print(f"1. IC-LoRA Adapter     : {'[OK] ' + str(paths.ic_lora_path) if paths.ic_lora_path else '[MISSING]'}")
    print(f"2. Scene Embedding     : {'[OK] ' + str(paths.scene_emb_path) if paths.scene_emb_path else '[MISSING]'}")
    print(f"3. Base Transformer    : {'[OK] ' + str(paths.transformer_path) if paths.transformer_path else '[MISSING]'}")
    print(f"4. Video VAE           : {'[OK] ' + str(paths.video_vae_path) if paths.video_vae_path else '[MISSING]'}")
    print("-" * 60)
    if paths.is_ready:
        print(">> Status: All components are present and ready for HDR conversion!")
    else:
        print(">> Status: Missing components detected:")
        for c in paths.missing_components or []:
            print(f"   - {c}")
        print("\nTo download missing base models from Hugging Face:")
        print("1. Ensure you have accepted terms at https://huggingface.co/Lightricks/LTX-2.5")
        print("2. Run (e.g. to E: drive):")
        print("   python scripts/download_models.py --download-base --dest-dir E:/models/LTX-2.5")
    print("=" * 60)


def download_base(token: str = None, dest_dir: str = None) -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Error: huggingface_hub is required. Install via pip install huggingface_hub")
        return

    dest_args = {}
    if dest_dir:
        dest_path = Path(dest_dir)
        dest_path.mkdir(parents=True, exist_ok=True)
        dest_args["local_dir"] = str(dest_path)
        print(f"Download destination target: {dest_path}")

    print("Downloading LTX-2.5 Video VAE (~1.5 GB)...")
    vae_file = hf_hub_download(
        repo_id=HF_BASE_REPO,
        filename=DEFAULT_VAE_FILENAME,
        token=token,
        **dest_args,
    )
    print(f"Video VAE downloaded: {vae_file}")

    print("Downloading LTX-2.5 22B Distilled Transformer (~42 GB)...")
    print("Note: This is a large 22B model. Ensure you have sufficient disk space.")
    trans_file = hf_hub_download(
        repo_id=HF_BASE_REPO,
        filename=DEFAULT_TRANSFORMER_FILENAME,
        token=token,
        **dest_args,
    )
    print(f"Transformer downloaded: {trans_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LTX-2.5 SDR-to-HDR Model Manager")
    parser.add_argument("--status", action="store_true", help="Check local model status")
    parser.add_argument("--download-base", action="store_true", help="Download base LTX-2.5 model files")
    parser.add_argument("--dest-dir", default=None, help="Directory to save downloaded model files (e.g. E:/models/LTX-2.5)")
    parser.add_argument("--token", default=None, help="Hugging Face access token")
    args = parser.parse_args()

    if args.download_base:
        download_base(token=args.token, dest_dir=args.dest_dir)
        print_status(custom_dir=args.dest_dir)
    else:
        print_status(custom_dir=args.dest_dir)


if __name__ == "__main__":
    main()
