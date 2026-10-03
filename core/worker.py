"""
core/worker.py
Headless CLI worker for executing LTX-2.5 SDR-to-HDR with CUDA acceleration.
Can be invoked by any Python environment (e.g. ComfyUI CUDA venv or direct CLI).
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add core parent directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.hdr_engine import HDRInferenceConfig, LTXHDREngine
from core.model_manager import LTXModelManager


def main() -> None:
    parser = argparse.ArgumentParser(description="LTX-2.5 SDR-To-HDR Worker")
    parser.add_argument("--input", required=True, help="Input SDR video path")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--colorspace", default="srgb_gamma", help="Input color space")
    parser.add_argument("--keyframe-strength", type=float, default=0.95, help="Keyframe strength")
    parser.add_argument("--quantization", default="fp8-cast", help="Quantization mode")
    parser.add_argument("--steps", type=int, default=8, help="Inference steps")
    parser.add_argument("--max-frames", type=int, default=0, help="Maximum frames to process")
    parser.add_argument("--custom-model-dir", default=None, help="Custom model directory to search")

    args = parser.parse_args()

    config = HDRInferenceConfig(
        input_colorspace=args.colorspace,
        keyframe_strength=args.keyframe_strength,
        num_inference_steps=args.steps,
        quantization=args.quantization,
        output_dir=args.output_dir,
        max_frames=args.max_frames,
    )

    model_paths = LTXModelManager.resolve_all_paths(custom_model_dir=args.custom_model_dir)
    engine = LTXHDREngine(config=config, model_paths=model_paths)

    def on_progress(p: float, msg: str) -> None:
        sys.stderr.write(f"[{int(p*100)}%] {msg}\n")
        sys.stderr.flush()

    res = engine.run_direct_inference(
        input_video_path=args.input,
        output_dir=args.output_dir,
        progress_callback=on_progress,
    )

    result_payload = {
        "was_successful": res.was_successful,
        "status_message": res.status_message,
        "exr_sequence_dir": res.exr_sequence_dir,
        "hlg_video_path": res.hlg_video_path,
        "preview_video_path": res.preview_video_path,
        "metadata_json_path": res.metadata_json_path,
        "total_frames": res.total_frames,
        "framerate": res.framerate,
        "source_colorspace": res.source_colorspace,
        "target_exr_colorspace": res.target_exr_colorspace,
        "target_hlg_colorspace": res.target_hlg_colorspace,
        "duration_seconds": res.duration_seconds,
        "device_used": res.device_used,
    }

    # Print final JSON result on stdout for parent process parsing
    print(json.dumps(result_payload))


if __name__ == "__main__":
    main()
