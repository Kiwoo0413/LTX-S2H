"""
core/hdr_engine.py
LTX-2.5 22B IC-LoRA SDR-To-HDR Inference Engine.

Implements the official Lightricks IC-LoRA architecture:
1. Frozen SDR full-clip conditioning reference at strength 1.0
2. Seam keyframe guides held at 0.95 strength
3. 8-step distilled Euler denoising schedule (zero CFG, zero spatial guidance)
4. Joint VAE latent decoding into ACEScct
5. ACEScct -> scene-linear ACEScg reconstruction
6. FP8 downcasting & sequential CPU offload support for 16GB GPUs (RTX 4080)
"""

from __future__ import annotations

import gc
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from core.io_utils import (
    ColorSpace,
    ColorSpaceConverter,
    EXRSequenceIO,
    VideoIO,
)
from core.model_manager import LTXModelManager, LTXModelPaths

logger = logging.getLogger("LTX_HDREngine")


@dataclass
class HDRInferenceConfig:
    """Inference parameters for LTX SDR-To-HDR."""
    input_colorspace: str = "srgb_gamma"
    keyframe_strength: float = 0.95
    num_inference_steps: int = 8
    quantization: str = "fp8-cast"  # "fp8-cast", "none", or "fp8-scaled-mm"
    enable_cpu_offload: bool = True
    seed: int = 42
    export_exr: bool = True
    export_hlg: bool = True
    export_preview_mp4: bool = True
    output_dir: Optional[str] = None
    max_frames: int = 0  # 0 = full video
    tiling: bool = True
    custom_model_dir: Optional[str] = None


@dataclass
class HDRInferenceResult:
    """Result of SDR-to-HDR processing."""
    was_successful: bool
    status_message: str
    exr_sequence_dir: Optional[str] = None
    hlg_video_path: Optional[str] = None
    preview_video_path: Optional[str] = None
    metadata_json_path: Optional[str] = None
    total_frames: int = 0
    framerate: float = 0.0
    source_colorspace: str = "srgb_gamma"
    target_exr_colorspace: str = "ACEScg"
    target_hlg_colorspace: str = "Rec.2100 HLG (BT.2020)"
    duration_seconds: float = 0.0
    device_used: str = "cpu"


class LTXHDREngine:
    """
    Core engine managing model loading, VRAM allocation, and SDR-To-HDR inference.
    Decoupled from Griptape host UI.
    """

    def __init__(
        self,
        config: Optional[HDRInferenceConfig] = None,
        model_paths: Optional[LTXModelPaths] = None,
    ) -> None:
        self.config = config or HDRInferenceConfig()
        self.model_paths = model_paths or LTXModelManager.resolve_all_paths()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def adjust_frame_count_for_ltx(frame_count: int) -> int:
        """
        LTX-2.5 requires temporal sequence length of 8k + 1 (e.g. 9, 17, 25, 33, 41, 49, 57, 65, 81, 97, 121...).
        Trims to largest 8k + 1 <= frame_count.
        """
        if frame_count < 9:
            return frame_count
        remainder = (frame_count - 1) % 8
        return frame_count - remainder

    def run_direct_inference(
        self,
        input_video_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> HDRInferenceResult:
        """
        Execute SDR-To-HDR inference in-process using PyTorch.
        """
        start_time = time.time()
        input_video_path = Path(input_video_path)
        if not input_video_path.exists():
            return HDRInferenceResult(
                was_successful=False,
                status_message=f"Input video not found: {input_video_path}"
            )

        # Output directory resolution
        output_dir = VideoIO.resolve_output_dir(input_video_path, custom_output_dir=output_dir, subfolder_suffix="HDR")
        video_stem = input_video_path.stem

        if progress_callback:
            progress_callback(0.05, "Extracting video frames and converting colorspace...")

        # 1. Video extraction
        frames_sdr, fps = VideoIO.extract_frames(
            input_video_path,
            max_frames=self.config.max_frames,
        )
        total_extracted = len(frames_sdr)
        valid_frames = self.adjust_frame_count_for_ltx(total_extracted)
        frames_sdr = frames_sdr[:valid_frames]

        logger.info(f"Loaded {valid_frames} frames @ {fps:.2f}fps for SDR-to-HDR processing.")

        if progress_callback:
            progress_callback(0.15, f"Transforming {valid_frames} frames to ACEScct working space...")

        # 2. Input transform to ACEScct
        acescct_input = ColorSpaceConverter.input_to_acescct(
            frames_sdr,
            source_space=self.config.input_colorspace
        )

        # 3. Model path verification & execution mode
        paths = self.model_paths
        is_neural = paths.is_ready
        fallback_notice = ""

        if not paths.is_ready:
            missing_str = ", ".join(paths.missing_components or [])
            fallback_notice = f"Missing base weights: {missing_str}"
            logger.warning(
                f"{fallback_notice}. Engaging mathematical ACES HDR highlight expansion fallback."
            )
        else:
            if progress_callback:
                progress_callback(0.25, "Loading LTX-2.5 IC-LoRA Pipeline with FP8 optimization...")

        # 4. Inference execution
        acescct_result = None

        if is_neural:
            try:
                from ltx_core.model.video_vae import AUTO_TILING
                from ltx_pipelines.hdr_ic_lora import HDRICLoraPipeline
                from ltx_pipelines.utils.media_io import VideoInput
                from ltx_pipelines.utils.model_paths import ModelPaths
                from ltx_pipelines.utils.quantization_factory import QuantizationKind
                from ltx_pipelines.utils.types import OffloadMode

                quant = QuantizationKind.FP8_CAST if "fp8" in str(self.config.quantization).lower() else None
                offload = OffloadMode.CPU if self.config.enable_cpu_offload else OffloadMode.NONE

                logger.info(f"Instantiating HDRICLoraPipeline (Quant: {quant}, Offload: {offload})...")
                pipeline = HDRICLoraPipeline(
                    model_paths=ModelPaths.from_split(
                        transformer_path=paths.transformer_path,
                        video_vae_path=paths.video_vae_path,
                    ),
                    hdr_lora=paths.ic_lora_path,
                    text_embeddings_path=paths.scene_emb_path,
                    quantization=quant,
                    offload_mode=offload,
                )

                if progress_callback:
                    progress_callback(0.40, "Executing 8-step distilled Euler denoising...")

                with torch.inference_mode():
                    acescct_hdr, out_fps = pipeline(
                        video=VideoInput(path=input_video_path, gamma_encoded=True),
                        seed=self.config.seed,
                        tiling_config=AUTO_TILING if self.config.tiling else None,
                        keyframe_strength=self.config.keyframe_strength,
                    )

                if isinstance(acescct_hdr, torch.Tensor):
                    acescct_result = acescct_hdr.detach().cpu().float().numpy()
                else:
                    acescct_result = np.array(acescct_hdr, dtype=np.float32)
                logger.info(f"Neural SDR-to-HDR inference completed. Output shape: {acescct_result.shape}")
            except Exception as e:
                logger.warning(f"Neural LTX-2.5 pipeline error ({e}); engaging algorithmic ACES HDR mapping.")
                logger.exception(e)
                is_neural = False

        if acescct_result is None:
            # Algorithmic ACES HDR highlight headroom expansion fallback
            logger.info("Executing mathematical ACES HDR highlight reconstruction.")
            linear_sdr = ColorSpaceConverter.srgb_gamma_to_linear(frames_sdr)
            expanded_lin = linear_sdr + 2.5 * np.maximum(linear_sdr - 0.75, 0.0)**1.8
            acescg_hdr = ColorSpaceConverter.srgb_linear_to_acescg(expanded_lin)
            acescct_result = ColorSpaceConverter.acescg_to_acescct(acescg_hdr)

        # 5. Convert ACEScct output back to scene-linear ACEScg
        if progress_callback:
            progress_callback(0.75, "Reconstructing scene-linear ACEScg color signal...")

        acescg_output = ColorSpaceConverter.acescct_to_acescg(acescct_result)

        # 6. Export files with embedded metadata
        exr_dir_path = None
        hlg_mp4_path = None
        preview_mp4_path = None
        source_colorspace = self.config.input_colorspace
        target_exr_cs = "ACEScg"
        target_hlg_cs = "Rec.2100 HLG (BT.2020)"

        if self.config.export_exr:
            if progress_callback:
                progress_callback(0.85, "Writing 16-bit half-float ACEScg EXR sequence with embedded metadata...")
            exr_dir = output_dir / "acescg_exr"
            EXRSequenceIO.write_sequence(
                exr_dir,
                acescg_output,
                prefix="hdr_",
                fps=fps,
                source_colorspace=source_colorspace,
                target_colorspace=target_exr_cs,
            )
            exr_dir_path = str(exr_dir.resolve())

        if self.config.export_hlg:
            if progress_callback:
                progress_callback(0.92, "Encoding 10-bit Rec.2100 HLG BT.2020 MP4 master...")
            hlg_file = output_dir / f"{video_stem}_HLG.mp4"
            VideoIO.encode_hlg_mp4(
                hlg_file,
                acescg_output,
                fps=fps,
                source_colorspace=source_colorspace,
            )
            hlg_mp4_path = str(hlg_file.resolve())

        if self.config.export_preview_mp4:
            if progress_callback:
                progress_callback(0.97, "Generating tonemapped desktop preview MP4...")
            preview_file = output_dir / f"{video_stem}_HDR_preview.mp4"
            VideoIO.encode_tonemapped_mp4(
                preview_file,
                acescg_output,
                fps=fps,
                source_colorspace=source_colorspace,
            )
            preview_mp4_path = str(preview_file.resolve())

        # 7. Write production metadata JSON sidecar
        metadata_json_file = output_dir / f"{video_stem}_metadata.json"
        metadata_payload = {
            "source_video_name": input_video_path.name,
            "source_video_path": str(input_video_path.resolve()),
            "source_framerate": float(fps),
            "source_colorspace": source_colorspace,
            "target_exr_colorspace": target_exr_cs,
            "target_hlg_colorspace": target_hlg_cs,
            "preview_colorspace": "sRGB Gamma (Tonemapped)",
            "total_frames_processed": valid_frames,
            "frame_width": int(frames_sdr[0].shape[1]) if len(frames_sdr) > 0 else 0,
            "frame_height": int(frames_sdr[0].shape[0]) if len(frames_sdr) > 0 else 0,
            "duration_seconds": float(valid_frames / fps) if fps > 0 else 0.0,
            "conversion_elapsed_seconds": round(time.time() - start_time, 2),
            "device_used": self.device,
            "pipeline": "LTX-2.5 22B IC-LoRA SDR-To-HDR",
            "quantization": self.config.quantization,
            "keyframe_strength": self.config.keyframe_strength,
            "inference_steps": self.config.num_inference_steps,
            "outputs": {
                "exr_sequence_dir": exr_dir_path,
                "hlg_video_path": hlg_mp4_path,
                "preview_video_path": preview_mp4_path,
            },
        }
        VideoIO.write_metadata_json(metadata_json_file, metadata_payload)
        metadata_json_path = str(metadata_json_file.resolve())

        elapsed = time.time() - start_time
        if progress_callback:
            progress_callback(1.0, f"Completed SDR to HDR conversion in {elapsed:.1f}s.")

        if is_neural:
            status_msg = f"Successfully generated Neural LTX-2.5 HDR deliverable ({valid_frames} frames in {elapsed:.1f}s)."
        else:
            status_msg = (
                f"[Algorithmic Fallback] Generated ACEScg EXRs & HLG MP4 ({valid_frames} frames in {elapsed:.1f}s). "
                f"Notice: {fallback_notice or 'Neural package uninitialized'}. Download base weights to activate Neural LTX-2.5 22B."
            )

        return HDRInferenceResult(
            was_successful=True,
            status_message=status_msg,
            exr_sequence_dir=exr_dir_path,
            hlg_video_path=hlg_mp4_path,
            preview_video_path=preview_mp4_path,
            metadata_json_path=metadata_json_path,
            total_frames=valid_frames,
            framerate=float(fps),
            source_colorspace=source_colorspace,
            target_exr_colorspace=target_exr_cs,
            target_hlg_colorspace=target_hlg_cs,
            duration_seconds=elapsed,
            device_used=self.device,
        )

    def execute(
        self,
        input_video_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> HDRInferenceResult:
        """
        Orchestrates inference. If running in a CPU-only host environment but a CUDA Python exists
        (like ComfyUI venv), dispatches execution to worker script for maximum RTX 4080 GPU acceleration.
        """
        cuda_py = LTXModelManager.get_cuda_python_executable()
        is_current_cuda = torch.cuda.is_available()

        # If current process has CUDA or no external CUDA venv exists, run directly
        if is_current_cuda or cuda_py == sys.executable:
            return self.run_direct_inference(input_video_path, output_dir, progress_callback)

        # Dispatch to external worker with CUDA Python
        logger.info(f"Dispatching LTX SDR-to-HDR execution to CUDA Python: {cuda_py}")
        worker_script = Path(__file__).resolve().parent / "worker.py"

        if progress_callback:
            progress_callback(0.1, f"Launching CUDA Worker on RTX 4080 ({Path(cuda_py).parent.parent.name})...")

        cmd = [
            cuda_py,
            str(worker_script),
            "--input", str(input_video_path),
            "--colorspace", self.config.input_colorspace,
            "--keyframe-strength", str(self.config.keyframe_strength),
            "--quantization", self.config.quantization,
            "--steps", str(self.config.num_inference_steps),
        ]
        if output_dir:
            cmd.extend(["--output-dir", str(output_dir)])
        if self.config.max_frames > 0:
            cmd.extend(["--max-frames", str(self.config.max_frames)])
        if self.config.custom_model_dir:
            cmd.extend(["--custom-model-dir", str(self.config.custom_model_dir)])

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            result_json = None
            for line in proc.stdout:
                line_str = line.strip()
                if line_str.startswith("{") and line_str.endswith("}"):
                    try:
                        result_json = json.loads(line_str)
                    except Exception:
                        pass
                elif line_str:
                    logger.info(f"[Worker]: {line_str}")

            proc.wait()

            if result_json and result_json.get("was_successful"):
                return HDRInferenceResult(
                    was_successful=True,
                    status_message=result_json.get("status_message", "Completed via CUDA Worker"),
                    exr_sequence_dir=result_json.get("exr_sequence_dir"),
                    hlg_video_path=result_json.get("hlg_video_path"),
                    preview_video_path=result_json.get("preview_video_path"),
                    metadata_json_path=result_json.get("metadata_json_path"),
                    total_frames=result_json.get("total_frames", 0),
                    framerate=float(result_json.get("framerate", 0.0)),
                    source_colorspace=result_json.get("source_colorspace", self.config.input_colorspace),
                    target_exr_colorspace=result_json.get("target_exr_colorspace", "ACEScg"),
                    target_hlg_colorspace=result_json.get("target_hlg_colorspace", "Rec.2100 HLG (BT.2020)"),
                    duration_seconds=result_json.get("duration_seconds", 0.0),
                    device_used="cuda (worker)",
                )
        except Exception as e:
            logger.warning(f"Worker dispatch failed ({e}), falling back to direct in-process execution.")

        return self.run_direct_inference(input_video_path, output_dir, progress_callback)
