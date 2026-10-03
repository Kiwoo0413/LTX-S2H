"""
nodes/ltx_sdr_to_hdr_node.py
LTX-2.5 22B IC-LoRA SDR-To-HDR Node for Griptape Nodes Desktop.

Converts standard 8-bit SDR video into 16-bit High Dynamic Range (HDR) video deliverables:
- Scene-linear ACEScg 16-bit float OpenEXR sequence for VFX/Nuke/Resolve
- 10-bit Rec.2100 HLG BT.2020 HEVC MP4 master for HDR displays
- 8-bit tonemapped preview video for desktop playback
- Optimized for NVIDIA RTX 4080 (16GB VRAM) via FP8 quantization and CPU offload
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from core.hdr_engine import HDRInferenceConfig, LTXHDREngine
from core.io_utils import VideoIO
from core.model_manager import LTXModelManager
from nodes.griptape_compat import DataNode, Parameter, ParameterMode

logger = logging.getLogger("LTX_SDR_To_HDR_Node")


def extract_video_path(val: Any) -> str:
    """Extract string filesystem path from str, Path, or Griptape VideoUrlArtifact."""
    if val is None:
        return ""
    if isinstance(val, (str, Path)):
        s = str(val).strip()
        # Handle Griptape path macros like {inputs}/videos/...
        if s.startswith("{inputs}"):
            workspace = LTXModelManager.get_workspace_root()
            s = s.replace("{inputs}", str(workspace / "inputs"))
        return s
    if hasattr(val, "value"):
        return extract_video_path(val.value)
    return str(val)


class LTXSDRToHDRNode(DataNode):
    """
    LTX-2.5 22B IC-LoRA SDR-To-HDR Conversion Node
    
    Transforms 8-bit SDR footage into 16-bit HDR using Lightricks' LTX-2.5 foundation video
    diffusion model and distilled 8-step IC-LoRA adapter.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # ── Inputs ───────────────────────────────────────────────────────────
        self.add_parameter(
            Parameter(
                name="input_video",
                type="str",
                default_value="",
                tooltip="Path to input SDR video file (MP4, MOV, ProRes, etc.)",
                display_name="Input SDR Video",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="input_colorspace",
                type="str",
                default_value="srgb_gamma",
                tooltip="Input color transform: 'srgb_gamma' (standard SDR), 'srgb' (linear), 'acescg', or 'acescct'",
                display_name="Input Colorspace",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="keyframe_strength",
                type="float",
                default_value=0.95,
                tooltip="Seam guide anchor strength (trained default 0.95 keeps conversion faithful to SDR content)",
                display_name="Keyframe Guide Strength",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="quantization",
                type="str",
                default_value="fp8-cast",
                tooltip="Precision mode: 'fp8-cast' (recommended for 16GB VRAM GPUs like RTX 4080), 'none' (BF16), or 'fp8-scaled-mm'",
                display_name="VRAM Quantization",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="inference_steps",
                type="int",
                default_value=8,
                tooltip="Euler denoising steps (LTX distilled IC-LoRA is optimized for 8 steps)",
                display_name="Inference Steps",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="export_exr",
                type="bool",
                default_value=True,
                tooltip="Export 16-bit half-float ACEScg OpenEXR sequence for VFX/Color Grading",
                display_name="Export ACEScg EXR Sequence",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="export_hlg",
                type="bool",
                default_value=True,
                tooltip="Export 10-bit Rec.2100 HLG BT.2020 HEVC MP4 master for direct HDR display",
                display_name="Export 10-bit HLG MP4",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="export_preview_mp4",
                type="bool",
                default_value=True,
                tooltip="Export tonemapped 8-bit sRGB MP4 for preview playback on standard monitors",
                display_name="Export Preview MP4",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="output_dir",
                type="str",
                default_value="",
                tooltip="Custom destination directory (if empty, defaults to '<video_name>_HDR' inside video directory)",
                display_name="Output Directory (Optional)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="max_frames",
                type="int",
                default_value=0,
                tooltip="Max frames to process (0 = entire video)",
                display_name="Max Frames",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="custom_model_dir",
                type="str",
                default_value="",
                tooltip="Optional custom path to model weights folder (e.g. 'E:/models/LTX-2.5' or 'E:/models')",
                display_name="Custom Model Directory (Optional)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )

        # ── Outputs ──────────────────────────────────────────────────────────
        self.add_parameter(
            Parameter(
                name="exr_sequence_dir",
                type="str",
                default_value="",
                tooltip="Directory containing the exported 16-bit ACEScg OpenEXR frames",
                display_name="EXR Sequence Directory",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )
        self.add_parameter(
            Parameter(
                name="hlg_video_path",
                type="str",
                default_value="",
                tooltip="Filesystem path of the generated 10-bit Rec.2100 HLG BT.2020 MP4 master",
                display_name="HLG Master Video Path",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )
        self.add_parameter(
            Parameter(
                name="preview_video_path",
                type="str",
                default_value="",
                tooltip="Filesystem path of the tonemapped SDR preview video",
                display_name="Preview Video Path",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )
        self.add_parameter(
            Parameter(
                name="frame_count",
                type="int",
                default_value=0,
                tooltip="Total number of frames processed",
                display_name="Processed Frames",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )
        self.add_parameter(
            Parameter(
                name="framerate",
                type="float",
                default_value=0.0,
                tooltip="Original source video frame rate (fps)",
                display_name="Source Framerate (fps)",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )
        self.add_parameter(
            Parameter(
                name="source_colorspace",
                type="str",
                default_value="srgb_gamma",
                tooltip="Color space of the source video",
                display_name="Source ColorSpace",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )
        self.add_parameter(
            Parameter(
                name="target_colorspace",
                type="str",
                default_value="ACEScg / Rec.2100 HLG",
                tooltip="Target HDR color spaces produced (ACEScg EXR & Rec.2100 HLG MP4)",
                display_name="Target ColorSpace",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )
        self.add_parameter(
            Parameter(
                name="metadata_json_path",
                type="str",
                default_value="",
                tooltip="Filesystem path of the conversion metadata JSON sidecar",
                display_name="Metadata JSON Path",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )
        self.add_parameter(
            Parameter(
                name="status",
                type="str",
                default_value="Ready",
                tooltip="Execution status and hardware device details",
                display_name="Status",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

    def process(self) -> None:
        """Execute SDR to HDR workflow when triggered in Griptape."""
        raw_video = self.get_parameter_value("input_video")
        video_path_str = extract_video_path(raw_video)

        if not video_path_str or not Path(video_path_str).exists():
            msg = f"Error: Input video path is invalid or file does not exist: '{video_path_str}'"
            self.set_parameter_value("status", msg)
            logger.error(msg)
            return

        input_colorspace = self.get_parameter_value("input_colorspace") or "srgb_gamma"
        keyframe_strength = float(self.get_parameter_value("keyframe_strength") or 0.95)
        quantization = self.get_parameter_value("quantization") or "fp8-cast"
        steps = int(self.get_parameter_value("inference_steps") or 8)
        export_exr = bool(self.get_parameter_value("export_exr"))
        export_hlg = bool(self.get_parameter_value("export_hlg"))
        export_preview = bool(self.get_parameter_value("export_preview_mp4"))
        custom_out_dir = self.get_parameter_value("output_dir") or None
        max_frames = int(self.get_parameter_value("max_frames") or 0)

        # Dynamic Output Directory: automatically inside source video's folder
        base_out_dir = VideoIO.resolve_output_dir(video_path_str, custom_output_dir=custom_out_dir, subfolder_suffix="HDR")
        video_stem = Path(video_path_str).stem

        exr_dir = base_out_dir / "acescg_exr"
        hlg_path = base_out_dir / f"{video_stem}_HLG.mp4"
        preview_path = base_out_dir / f"{video_stem}_HDR_preview.mp4"

        exr_dir.mkdir(parents=True, exist_ok=True)

        # Immediately populate dynamic paths into node parameters so UI and downstream nodes reflect them
        self.set_parameter_value("output_dir", str(base_out_dir))
        self.set_parameter_value("exr_sequence_dir", str(exr_dir))
        self.set_parameter_value("hlg_video_path", str(hlg_path))
        self.set_parameter_value("preview_video_path", str(preview_path))

        custom_model_dir = self.get_parameter_value("custom_model_dir") or None

        config = HDRInferenceConfig(
            input_colorspace=input_colorspace,
            keyframe_strength=keyframe_strength,
            num_inference_steps=steps,
            quantization=quantization,
            export_exr=export_exr,
            export_hlg=export_hlg,
            export_preview_mp4=export_preview,
            output_dir=str(base_out_dir),
            max_frames=max_frames,
            custom_model_dir=custom_model_dir,
        )

        model_paths = LTXModelManager.resolve_all_paths(custom_model_dir=custom_model_dir)
        engine = LTXHDREngine(config=config, model_paths=model_paths)

        self.set_parameter_value("status", "Processing SDR to HDR conversion...")

        res = engine.execute(
            input_video_path=video_path_str,
            output_dir=str(base_out_dir),
            progress_callback=lambda p, msg: self.set_parameter_value("status", f"[{int(p*100)}%] {msg}"),
        )

        self.set_parameter_value("exr_sequence_dir", res.exr_sequence_dir or str(exr_dir))
        self.set_parameter_value("hlg_video_path", res.hlg_video_path or str(hlg_path))
        self.set_parameter_value("preview_video_path", res.preview_video_path or str(preview_path))
        self.set_parameter_value("metadata_json_path", res.metadata_json_path or str(base_out_dir / f"{video_stem}_metadata.json"))
        self.set_parameter_value("framerate", float(res.framerate))
        self.set_parameter_value("source_colorspace", str(res.source_colorspace))
        self.set_parameter_value("target_colorspace", f"{res.target_exr_colorspace} / {res.target_hlg_colorspace}")
        self.set_parameter_value("frame_count", res.total_frames)
        self.set_parameter_value("status", res.status_message)
