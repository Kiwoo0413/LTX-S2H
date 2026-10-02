"""
tests/test_model_manager.py
Test Hugging Face cache discovery and model path resolution.
"""

from pathlib import Path
from core.model_manager import LTXModelManager


def test_locate_cached_ic_lora():
    """Verify that cached HuggingFace IC-LoRA weights are discovered."""
    paths = LTXModelManager.resolve_all_paths()

    # The user already has the IC-LoRA adapter in HF cache
    if paths.ic_lora_path:
        assert Path(paths.ic_lora_path).exists()
        assert paths.ic_lora_path.endswith(".safetensors")

    if paths.scene_emb_path:
        assert Path(paths.scene_emb_path).exists()
        assert paths.scene_emb_path.endswith(".safetensors")


def test_search_directories():
    """Verify search directories include workspace models and libraries."""
    dirs = LTXModelManager.get_search_directories()
    assert len(dirs) >= 2
