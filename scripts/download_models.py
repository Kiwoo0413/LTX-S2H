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


REPO_VAE_PATH = "vae/ltx-2.5-video-vae-bf16.safetensors"
REPO_TRANSFORMER_PATH = "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"


def download_base(token: str = None, dest_dir: str = None, download_vae: bool = True, download_trans: bool = True) -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Error: huggingface_hub is required. Install via pip install huggingface_hub")
        return

    dest_args = {}
    dest_path = Path(dest_dir) if dest_dir else None
    if dest_path:
        dest_path.mkdir(parents=True, exist_ok=True)
        dest_args["local_dir"] = str(dest_path)
        print(f"Download destination target: {dest_path}")

    if download_vae:
        local_vae = dest_path / REPO_VAE_PATH if dest_path else None
        if local_vae and local_vae.exists() and local_vae.stat().st_size > 1_000_000_000:
            print(f"Video VAE already present ({local_vae.stat().st_size / (1024**3):.2f} GB). Skipping VAE download.")
        else:
            print("Downloading LTX-2.5 Video VAE (~1.5 GB)...")
            vae_file = hf_hub_download(
                repo_id=HF_BASE_REPO,
                filename=REPO_VAE_PATH,
                token=token,
                **dest_args,
            )
            print(f"Video VAE ready: {vae_file}")

    if download_trans:
        local_trans = dest_path / REPO_TRANSFORMER_PATH if dest_path else None
        if local_trans and local_trans.exists() and local_trans.stat().st_size > 35_000_000_000:
            print(f"Transformer already present ({local_trans.stat().st_size / (1024**3):.2f} GB). Skipping download.")
        else:
            print("Downloading LTX-2.5 22B Distilled Transformer (~39.1 GB)...")
            print("Note: This is a large 22B model. It may take some time depending on bandwidth.")
            trans_file = None
            for repo in [HF_BASE_REPO, "comfyicu/LTX-2.5"]:
                try:
                    print(f"Attempting download from repo: {repo}...")
                    trans_file = hf_hub_download(
                        repo_id=repo,
                        filename=REPO_TRANSFORMER_PATH,
                        token=token,
                        **dest_args,
                    )
                    break
                except Exception as e:
                    print(f"Failed to download from {repo}: {e}")
            if trans_file:
                print(f"Transformer ready: {trans_file}")
            else:
                raise RuntimeError("Failed to download transformer from all candidate repositories.")


def main() -> None:
    parser = argparse.ArgumentParser(description="LTX-2.5 SDR-to-HDR Model Manager")
    parser.add_argument("--status", action="store_true", help="Check local model status")
    parser.add_argument("--download-base", action="store_true", help="Download base LTX-2.5 model files")
    parser.add_argument("--download-transformer", action="store_true", help="Download only the distilled transformer")
    parser.add_argument("--download-vae", action="store_true", help="Download only the video VAE")
    parser.add_argument("--dest-dir", default="E:/models/LTX-2.5", help="Directory to save downloaded model files (default: E:/models/LTX-2.5)")
    parser.add_argument("--token", default=None, help="Hugging Face access token (optional, auto-read from cache)")
    args = parser.parse_args()

    if args.download_base or args.download_transformer or args.download_vae:
        download_vae = args.download_base or args.download_vae
        download_trans = args.download_base or args.download_transformer
        download_base(token=args.token, dest_dir=args.dest_dir, download_vae=download_vae, download_trans=download_trans)
        print_status(custom_dir=args.dest_dir)
    else:
        print_status(custom_dir=args.dest_dir)


if __name__ == "__main__":
    main()
