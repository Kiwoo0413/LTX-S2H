"""
core/model_manager.py
Model Discovery, Hugging Face Cache Resolution, and Environment Detection for LTX-2.5 SDR-To-HDR.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("LTX_ModelManager")

HF_IC_LORA_REPO = "Lightricks/LTX-2.5-22b-IC-LoRA-SDR-To-HDR"
HF_BASE_REPO = "Lightricks/LTX-2.5"

DEFAULT_LORA_FILENAME = "ltx-2.5-22b-ic-lora-sdr-to-hdr-1.0.safetensors"
DEFAULT_SCENE_EMB_FILENAME = "ltx-2.5-22b-ic-lora-sdr-to-hdr-scene-emb.safetensors"
DEFAULT_TRANSFORMER_FILENAME = "ltx-2.5-22b-distilled-transformer-bf16.safetensors"
DEFAULT_VAE_FILENAME = "ltx-2.5-video-vae-bf16.safetensors"


@dataclass
class LTXModelPaths:
    """Resolved file paths for all required LTX-2.5 components."""
    ic_lora_path: Optional[str] = None
    scene_emb_path: Optional[str] = None
    transformer_path: Optional[str] = None
    video_vae_path: Optional[str] = None
    is_ready: bool = False
    missing_components: Optional[List[str]] = None


class LTXModelManager:
    """Manages weights discovery across Hugging Face cache and workspace directories."""

    @staticmethod
    def get_workspace_root() -> Path:
        """Dynamically detect workspace root by ascending from library root."""
        lib_root = Path(__file__).resolve().parent.parent
        if lib_root.parent.name == "libraries":
            return lib_root.parent.parent
        return lib_root.parent

    @classmethod
    def get_search_directories(cls) -> List[Path]:
        """Candidate search directories for model weights."""
        dirs = []
        workspace = cls.get_workspace_root()
        lib_root = Path(__file__).resolve().parent.parent

        # 1. Library models dir
        dirs.append(lib_root / "models")
        # 2. Workspace models dir
        dirs.append(workspace / "models")
        # 3. ComfyUI models dir if exists
        comfy_models = Path("D:/AI/ComfyUI_DATA/models")
        if comfy_models.exists():
            dirs.append(comfy_models / "diffusion_models")
            dirs.append(comfy_models / "loras")
            dirs.append(comfy_models / "vae")

        return dirs

    @classmethod
    def find_in_hf_cache(cls, repo_id: str, filename: str) -> Optional[str]:
        """Search Hugging Face hub cache snapshots for the specified repo and file."""
        repo_folder_name = "models--" + repo_id.replace("/", "--")
        cache_hub = Path.home() / ".cache" / "huggingface" / "hub" / repo_folder_name / "snapshots"

        if not cache_hub.exists():
            return None

        # Check all snapshot revisions (newest first)
        snapshots = sorted(cache_hub.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        for snap in snapshots:
            target = snap / filename
            if target.is_file():
                return str(target.resolve())

        return None

    @classmethod
    def locate_file(cls, filename: str, repo_id: Optional[str] = None) -> Optional[str]:
        """Locate file in HF cache or search directories."""
        # 1. Search in Hugging Face cache
        if repo_id:
            hf_path = cls.find_in_hf_cache(repo_id, filename)
            if hf_path:
                return hf_path

        # 2. Search search directories
        for search_dir in cls.get_search_directories():
            if not search_dir.exists():
                continue
            cand = search_dir / filename
            if cand.is_file():
                return str(cand.resolve())
            # Search subdirectories 1 level
            for sub in search_dir.glob(f"**/{filename}"):
                if sub.is_file():
                    return str(sub.resolve())

        return None

    @classmethod
    def resolve_all_paths(
        cls,
        custom_lora: Optional[str] = None,
        custom_scene_emb: Optional[str] = None,
        custom_transformer: Optional[str] = None,
        custom_vae: Optional[str] = None,
    ) -> LTXModelPaths:
        """Resolve paths for all components, returning status and list of missing parts."""
        lora_path = custom_lora or cls.locate_file(DEFAULT_LORA_FILENAME, HF_IC_LORA_REPO)
        scene_emb_path = custom_scene_emb or cls.locate_file(DEFAULT_SCENE_EMB_FILENAME, HF_IC_LORA_REPO)
        transformer_path = custom_transformer or cls.locate_file(DEFAULT_TRANSFORMER_FILENAME, HF_BASE_REPO)
        vae_path = custom_vae or cls.locate_file(DEFAULT_VAE_FILENAME, HF_BASE_REPO)

        missing = []
        if not lora_path or not os.path.exists(lora_path):
            missing.append(f"IC-LoRA Adapter ({DEFAULT_LORA_FILENAME})")
        if not scene_emb_path or not os.path.exists(scene_emb_path):
            missing.append(f"Scene Embedding ({DEFAULT_SCENE_EMB_FILENAME})")
        if not transformer_path or not os.path.exists(transformer_path):
            missing.append(f"Base Distilled Transformer ({DEFAULT_TRANSFORMER_FILENAME})")
        if not vae_path or not os.path.exists(vae_path):
            missing.append(f"Video VAE ({DEFAULT_VAE_FILENAME})")

        return LTXModelPaths(
            ic_lora_path=lora_path,
            scene_emb_path=scene_emb_path,
            transformer_path=transformer_path,
            video_vae_path=vae_path,
            is_ready=len(missing) == 0,
            missing_components=missing,
        )

    @staticmethod
    def get_cuda_python_executable() -> str:
        """Find Python executable with CUDA PyTorch support."""
        # 1. Current runtime if CUDA available
        try:
            import torch
            if torch.cuda.is_available():
                return sys.executable
        except Exception:
            pass

        # 2. Known local CUDA venvs (e.g. ComfyUI venv on D:\AI)
        candidate_venvs = [
            Path("D:/AI/ComfyUI_DATA/.venv/Scripts/python.exe"),
            Path("D:/AI/ComfyUI/.venv/Scripts/python.exe"),
        ]
        for py in candidate_venvs:
            if py.is_file():
                return str(py.resolve())

        # Fallback to current sys.executable
        return sys.executable
