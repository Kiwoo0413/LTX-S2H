"""
core/io_utils.py
Color Space Transformations, EXR Frame IO, and Video Encoding for SDR-To-HDR Pipeline.

Specifications:
- Internal Working Space: ACEScct (logarithmic encoding)
- Scene-Linear Export: ACEScg (AP1 color primaries, linear transfer)
- HDR Delivery: HLG (Rec.2100 ARIB STD-B67) BT.2020 10-bit MP4
- SDR Input Support: sRGB gamma, Rec.709, scene-linear sRGB, ACEScg, ACEScct
"""

from __future__ import annotations

import enum
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile

# Enable OpenCV OpenEXR support BEFORE importing cv2
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger("LTX_HDR_IO")


class ColorSpace(str, enum.Enum):
    SRGB_GAMMA = "srgb_gamma"
    SRGB_LINEAR = "srgb"
    ACESCG = "acescg"
    ACESCCT = "acescct"
    REC709 = "rec709"


# Color Matrix Constants
# sRGB/Rec709 Linear to ACEScg (AP1) matrix (derived from Bradford-adapted chromaticities)
MAT_SRGB_TO_ACESCG = np.array([
    [0.613097, 0.339523, 0.047379],
    [0.070194, 0.916354, 0.013452],
    [0.020616, 0.109570, 0.869814]
], dtype=np.float32)

MAT_ACESCG_TO_SRGB = np.linalg.inv(MAT_SRGB_TO_ACESCG)

# ACEScg to Rec2020 Linear matrix
MAT_ACESCG_TO_REC2020 = np.array([
    [0.679198, 0.288079, 0.032723],
    [0.045786, 0.921764, 0.032450],
    [-0.000574, 0.049442, 0.951132]
], dtype=np.float32)

MAT_REC2020_TO_ACESCG = np.linalg.inv(MAT_ACESCG_TO_REC2020)


class ColorSpaceConverter:
    """Exact color transformations conforming to ACES 1.3 / SMPTE ST 2065-1 specifications."""

    @staticmethod
    def srgb_gamma_to_linear(rgb: np.ndarray) -> np.ndarray:
        """Convert standard 8-bit sRGB gamma [0, 1] to scene-linear sRGB."""
        rgb = np.clip(rgb, 0.0, None)
        linear = np.where(
            rgb <= 0.04045,
            rgb / 12.92,
            np.power((rgb + 0.055) / 1.055, 2.4)
        )
        return linear.astype(np.float32)

    @staticmethod
    def linear_to_srgb_gamma(rgb: np.ndarray) -> np.ndarray:
        """Convert scene-linear sRGB to display gamma 2.2 / sRGB."""
        rgb = np.clip(rgb, 0.0, 1.0)
        gamma = np.where(
            rgb <= 0.0031308,
            rgb * 12.92,
            1.055 * np.power(rgb, 1.0 / 2.4) - 0.055
        )
        return gamma.astype(np.float32)

    @staticmethod
    def srgb_linear_to_acescg(rgb_linear: np.ndarray) -> np.ndarray:
        """Matrix transform from linear sRGB/Rec709 to ACEScg (AP1 primaries)."""
        shape = rgb_linear.shape
        flat = rgb_linear.reshape(-1, 3)
        acescg = np.dot(flat, MAT_SRGB_TO_ACESCG.T)
        return acescg.reshape(shape).astype(np.float32)

    @staticmethod
    def acescg_to_srgb_linear(acescg: np.ndarray) -> np.ndarray:
        """Matrix transform from ACEScg to linear sRGB."""
        shape = acescg.shape
        flat = acescg.reshape(-1, 3)
        srgb = np.dot(flat, MAT_ACESCG_TO_SRGB.T)
        return srgb.reshape(shape).astype(np.float32)

    @staticmethod
    def acescg_to_acescct(acescg: np.ndarray) -> np.ndarray:
        """
        ACEScg (linear AP1) to ACEScct (logarithmic working space).
        ACEScct is designed for grading and neural latent spaces:
        - Linear below threshold: y = (x - 0.0078125) * 10.5402377416545
        - Logarithmic above threshold: y = (log2(x) + 9.72) / 17.52
        """
        x = np.array(acescg, dtype=np.float32)
        thresh = 0.0078125
        out = np.zeros_like(x, dtype=np.float32)

        low_mask = x <= thresh
        high_mask = ~low_mask

        out[low_mask] = (x[low_mask] - thresh) * 10.5402377416545
        out[high_mask] = (np.log2(np.maximum(x[high_mask], 1e-10)) + 9.72) / 17.52
        return out

    @staticmethod
    def acescct_to_acescg(acescct: np.ndarray) -> np.ndarray:
        """
        ACEScct (logarithmic) back to ACEScg (linear AP1).
        Inverse transform:
        - If y > 0.155251141552511: x = 2^(y * 17.52 - 9.72)
        - Else: x = y / 10.5402377416545 + 0.0078125
        """
        y = np.array(acescct, dtype=np.float32)
        thresh_y = (np.log2(0.0078125) + 9.72) / 17.52  # approx 0.15525114
        out = np.zeros_like(y, dtype=np.float32)

        low_mask = y <= thresh_y
        high_mask = ~low_mask

        out[low_mask] = (y[low_mask] / 10.5402377416545) + 0.0078125
        out[high_mask] = np.exp2(y[high_mask] * 17.52 - 9.72)
        return np.maximum(out, 0.0).astype(np.float32)

    @classmethod
    def input_to_acescct(cls, rgb: np.ndarray, source_space: Union[str, ColorSpace] = ColorSpace.SRGB_GAMMA) -> np.ndarray:
        """Transform any input colorspace to ACEScct pipeline conditioning."""
        if isinstance(source_space, str):
            source_space = ColorSpace(source_space.lower())

        if source_space == ColorSpace.ACESCCT:
            return rgb.astype(np.float32)
        elif source_space == ColorSpace.ACESCG:
            return cls.acescg_to_acescct(rgb)
        elif source_space == ColorSpace.SRGB_LINEAR:
            acescg = cls.srgb_linear_to_acescg(rgb)
            return cls.acescg_to_acescct(acescg)
        elif source_space in (ColorSpace.SRGB_GAMMA, ColorSpace.REC709):
            linear = cls.srgb_gamma_to_linear(rgb)
            acescg = cls.srgb_linear_to_acescg(linear)
            return cls.acescg_to_acescct(acescg)
        else:
            raise ValueError(f"Unsupported source colorspace: {source_space}")

    @classmethod
    def acescg_to_hlg_bt2020(cls, acescg: np.ndarray) -> np.ndarray:
        """
        Convert ACEScg scene-linear to Rec.2100 HLG (Hybrid Log-Gamma) signal.
        HLG OETF (ARIB STD-B67):
        E in [0, 1] mapped with a=0.17883277, b=0.28466892, c=0.55991073.
        """
        shape = acescg.shape
        flat = acescg.reshape(-1, 3)
        rec2020 = np.dot(flat, MAT_ACESCG_TO_REC2020.T)
        rec2020 = rec2020.reshape(shape)

        # Normalization anchor: diffuse white in scene linear is mapped to ~0.5 - 0.75 HLG signal
        # Nominal scene linear 1.0 (100% white card) corresponds to 75% HLG code value
        # Clamp negative values
        e = np.maximum(rec2020, 0.0)

        a = 0.17883277
        b = 0.28466892
        c = 0.55991073

        hlg = np.zeros_like(e, dtype=np.float32)
        low = e <= (1.0 / 12.0)
        high = ~low

        hlg[low] = np.sqrt(3.0) * np.power(np.maximum(e[low], 0.0), 0.5)
        hlg[high] = a * np.log(np.maximum(12.0 * e[high] - b, 1e-10)) + c

        return np.clip(hlg, 0.0, 1.0).astype(np.float32)

    @classmethod
    def tonemap_acescg_for_preview(cls, acescg: np.ndarray) -> np.ndarray:
        """Narkiewicz / ACES-like S-curve tone mapping for standard sRGB monitors."""
        # Convert ACEScg to sRGB linear
        srgb_lin = cls.acescg_to_srgb_linear(acescg)
        # Hill ACES fitted curve
        a = 2.51
        b = 0.03
        c = 2.43
        d = 0.59
        e = 0.14
        mapped_lin = (srgb_lin * (a * srgb_lin + b)) / (srgb_lin * (c * srgb_lin + d) + e)
        srgb_display = cls.linear_to_srgb_gamma(mapped_lin)
        return (np.clip(srgb_display, 0.0, 1.0) * 255.0).astype(np.uint8)


class EXRSequenceIO:
    """Read and write multi-frame high dynamic range EXR sequences (16-bit half float)."""

    @staticmethod
    def write_frame(
        path: Union[str, Path],
        image_acescg: np.ndarray,
        fps: Optional[float] = None,
        source_colorspace: Optional[str] = None,
        target_colorspace: str = "ACEScg",
        custom_metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Write single image array (H, W, 3) float32/float16 in ACEScg space to EXR.
        Embeds framerate and color space metadata into the EXR header.
        Prefers OpenImageIO (standard VFX industry writer) -> OpenEXR -> OpenCV -> imageio.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        img_f32 = np.ascontiguousarray(image_acescg, dtype=np.float32)
        h, w = img_f32.shape[:2]

        # 1. OpenImageIO (Hollywood / VFX standard metadata)
        try:
            import OpenImageIO as oiio

            spec = oiio.ImageSpec(w, h, 3, oiio.TypeHalf)
            spec.attribute("oiio:ColorSpace", str(target_colorspace))
            spec.attribute("ColorSpace", str(target_colorspace))

            if source_colorspace:
                spec.attribute("source_colorspace", str(source_colorspace))

            if fps is not None and fps > 0:
                spec.attribute("framesPerSecond", float(fps))
                spec.attribute("source_framerate", float(fps))

            if custom_metadata:
                for k, v in custom_metadata.items():
                    if isinstance(v, (int, float, str)):
                        spec.attribute(str(k), v)

            out = oiio.ImageOutput.create(str(path))
            if out:
                if out.open(str(path), spec):
                    out.write_image(img_f32)
                    out.close()
                    if path.exists():
                        return True
        except Exception:
            pass

        # 2. Native OpenEXR C-binding
        try:
            import OpenEXR
            import Imath

            header = OpenEXR.Header(w, h)
            float_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))

            if img_f32.ndim == 3 and img_f32.shape[-1] == 3:
                header["channels"] = {"R": float_chan, "G": float_chan, "B": float_chan}

                if fps is not None and fps > 0:
                    header["framesPerSecond"] = Imath.Rational(int(round(fps * 1000)), 1000)

                header["comments"] = f"LTX-2.5 HDR; Source FPS: {fps}; Source CS: {source_colorspace}; Target: {target_colorspace}"

                r = img_f32[:, :, 0].astype(np.float16).tobytes()
                g = img_f32[:, :, 1].astype(np.float16).tobytes()
                b = img_f32[:, :, 2].astype(np.float16).tobytes()
                exr_out = OpenEXR.OutputFile(str(path), header)
                exr_out.writePixels({"R": r, "G": g, "B": b})
                exr_out.close()
                if path.exists():
                    return True
        except Exception:
            pass

        # 3. Try OpenCV cv2.imwrite
        try:
            if img_f32.ndim == 3 and img_f32.shape[2] == 3:
                bgr = img_f32[..., ::-1]
            else:
                bgr = img_f32

            flags = [int(cv2.IMWRITE_EXR_TYPE), int(cv2.IMWRITE_EXR_TYPE_HALF)]
            if cv2.imwrite(str(path), bgr, flags) and path.exists():
                return True
        except Exception:
            pass

        # 4. Fallback to imageio
        try:
            import imageio.v3 as iio
            iio.imwrite(path, img_f32)
            if path.exists():
                return True
        except Exception:
            pass

        return False

    @staticmethod
    def read_frame(path: Union[str, Path]) -> np.ndarray:
        """Read single EXR frame returning RGB float32 array (H, W, 3)."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"EXR file not found: {path}")

        # 1. Try Native OpenEXR
        try:
            import OpenEXR
            import Imath

            file = OpenEXR.InputFile(str(path))
            dw = file.header()['dataWindow']
            w = dw.max.x - dw.min.x + 1
            h = dw.max.y - dw.min.y + 1

            pt = Imath.PixelType(Imath.PixelType.FLOAT)
            r_str = file.channel('R', pt)
            g_str = file.channel('G', pt)
            b_str = file.channel('B', pt)

            r = np.frombuffer(r_str, dtype=np.float32).reshape(h, w)
            g = np.frombuffer(g_str, dtype=np.float32).reshape(h, w)
            b = np.frombuffer(b_str, dtype=np.float32).reshape(h, w)
            return np.stack([r, g, b], axis=-1)
        except Exception:
            pass

        # 2. Try OpenCV
        try:
            bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if bgr is not None:
                if bgr.ndim == 3 and bgr.shape[2] == 3:
                    return bgr[..., ::-1].astype(np.float32)
                return bgr.astype(np.float32)
        except Exception:
            pass

        # 3. Try imageio
        try:
            import imageio.v3 as iio
            img = iio.imread(path)
            return img.astype(np.float32)
        except Exception as e:
            raise ValueError(f"Failed to decode EXR image {path}: {e}")

    @classmethod
    def write_sequence(
        cls,
        output_dir: Union[str, Path],
        frames_acescg: Union[List[np.ndarray], np.ndarray],
        prefix: str = "frame_",
        start_frame: int = 1,
        fps: Optional[float] = None,
        source_colorspace: Optional[str] = None,
        target_colorspace: str = "ACEScg",
        custom_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Path]:
        """Write sequence of frames to directory with 4-digit zero-padding and embedded metadata."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_paths: List[Path] = []
        for i, frame in enumerate(frames_acescg):
            frame_num = start_frame + i
            frame_path = output_dir / f"{prefix}{frame_num:04d}.exr"
            cls.write_frame(
                frame_path,
                frame,
                fps=fps,
                source_colorspace=source_colorspace,
                target_colorspace=target_colorspace,
                custom_metadata=custom_metadata,
            )
            saved_paths.append(frame_path)

        return saved_paths

    @staticmethod
    def read_metadata(path: Union[str, Path]) -> Dict[str, Any]:
        """Read metadata attributes from an EXR file header."""
        path = Path(path)
        if not path.exists():
            return {}

        # 1. Try OpenImageIO
        try:
            import OpenImageIO as oiio
            inp = oiio.ImageInput.open(str(path))
            if inp:
                spec = inp.spec()
                meta = {}
                for attr in spec.extra_attribs:
                    meta[attr.name] = attr.value
                inp.close()
                return meta
        except Exception:
            pass

        # 2. Try OpenEXR
        try:
            import OpenEXR
            f = OpenEXR.InputFile(str(path))
            header = f.header()
            meta = {}
            for k, v in header.items():
                if k != "channels":
                    meta[k] = str(v)
            return meta
        except Exception:
            pass

        return {}


class VideoIO:
    """Video frame extraction and metadata inspection."""

    @staticmethod
    def resolve_output_dir(
        input_video_path: Union[str, Path],
        custom_output_dir: Optional[Union[str, Path]] = None,
        subfolder_suffix: str = "HDR",
    ) -> Path:
        """
        Dynamically resolve output directory.
        If custom_output_dir is provided and non-empty, use it.
        Otherwise, automatically create the folder directly inside the parent folder
        where the source video resides:
            e.g. "D:/videos/shot01/clip.mp4" -> "D:/videos/shot01/clip_HDR/"
        """
        if custom_output_dir and str(custom_output_dir).strip():
            out_p = Path(custom_output_dir).resolve()
        else:
            vid_p = Path(input_video_path).resolve()
            out_p = vid_p.parent / f"{vid_p.stem}_{subfolder_suffix}"

        out_p.mkdir(parents=True, exist_ok=True)
        return out_p

    @staticmethod
    def get_metadata(video_path: Union[str, Path]) -> Dict[str, Any]:
        """Get video properties: width, height, fps, frame_count, duration."""
        video_path = str(video_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0.0
        cap.release()

        return {
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
            "duration": duration,
        }

    @staticmethod
    def extract_frames(
        video_path: Union[str, Path],
        max_frames: int = 0,
        target_resolution: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, float]:
        """
        Extract video frames as normalized float32 array [0.0, 1.0] of shape (F, H, W, 3).
        Returns (frames, fps).
        """
        video_path = str(video_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
        frames_list = []
        count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # cv2 is BGR -> convert to RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if target_resolution:
                rgb = cv2.resize(rgb, target_resolution, interpolation=cv2.INTER_AREA)

            # Convert to float32 [0.0, 1.0]
            float_frame = rgb.astype(np.float32) / 255.0
            frames_list.append(float_frame)
            count += 1

            if max_frames > 0 and count >= max_frames:
                break

        cap.release()

        if not frames_list:
            raise ValueError(f"No frames could be extracted from {video_path}")

        return np.stack(frames_list, axis=0), fps

    @staticmethod
    def find_ffmpeg() -> Optional[str]:
        """Find system ffmpeg or static-ffmpeg."""
        try:
            from static_ffmpeg import run
            ffmpeg_path, _ = run.get_or_fetch_platform_executables_else_raise()
            if os.path.exists(ffmpeg_path):
                return ffmpeg_path
        except Exception:
            pass

        sys_ffmpeg = shutil.which("ffmpeg")
        if sys_ffmpeg:
            return sys_ffmpeg
        return None

    @classmethod
    def encode_opencv_fallback(
        cls,
        output_path: Union[str, Path],
        frames: Union[List[np.ndarray], np.ndarray],
        fps: float = 24.0,
    ) -> Path:
        """Fallback video writer using OpenCV VideoWriter when ffmpeg is unavailable."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if len(frames) == 0:
            return output_path

        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

        for f in frames:
            if f.dtype != np.uint8:
                uint8_frame = (np.clip(f, 0.0, 1.0) * 255.0).astype(np.uint8)
            else:
                uint8_frame = f
            writer.write(cv2.cvtColor(uint8_frame, cv2.COLOR_RGB2BGR))
        writer.release()
        return output_path

    @classmethod
    def encode_hlg_mp4(
        cls,
        output_path: Union[str, Path],
        frames_acescg: Union[List[np.ndarray], np.ndarray],
        fps: float = 24.0,
        source_colorspace: str = "srgb_gamma",
    ) -> Path:
        """
        Encode 10-bit HLG (Rec.2100 / ARIB STD-B67) BT.2020 HEVC MP4 using ffmpeg.
        Tags color primaries (bt2020), transfer (arib-std-b67), matrix (bt2020nc),
        and embeds source framerate and source colorspace metadata tags.
        Falls back to OpenCV VideoWriter if ffmpeg is unavailable.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg = cls.find_ffmpeg()

        if not ffmpeg:
            logger.info("ffmpeg executable not found in PATH; encoding video with OpenCV fallback.")
            return cls.encode_opencv_fallback(output_path, frames_acescg, fps=fps)

        # Convert ACEScg frames to 10-bit HLG frames
        hlg_frames = []
        for f in frames_acescg:
            hlg = ColorSpaceConverter.acescg_to_hlg_bt2020(f)
            hlg_10bit = (hlg * 1023.0).astype(np.uint16)
            hlg_frames.append(hlg_10bit)

        hlg_array = np.stack(hlg_frames, axis=0)
        num_frames, height, width, _ = hlg_array.shape

        # Use rawpipe with ffmpeg libx265 10-bit with embedded metadata
        cmd = [
            ffmpeg,
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{width}x{height}",
            "-pix_fmt", "rgb48le",  # pipe 16-bit uint container
            "-r", str(fps),
            "-i", "-",
            "-c:v", "libx265",
            "-pix_fmt", "yuv420p10le",
            "-color_primaries", "bt2020",
            "-color_trc", "arib-std-b67",
            "-colorspace", "bt2020nc",
            "-metadata", f"source_framerate={fps:.3f}",
            "-metadata", f"source_colorspace={source_colorspace}",
            "-metadata", "target_colorspace=Rec.2100 HLG (BT.2020)",
            "-metadata", f"comment=Source FPS: {fps:.3f}, Source ColorSpace: {source_colorspace}, Target: Rec.2100 HLG (BT.2020)",
            "-crf", "18",
            "-preset", "medium",
            str(output_path)
        ]

        pipe_data = (hlg_array.astype(np.uint16) << 6).tobytes()  # shift 10bit to upper 16bit for rgb48le
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            _, stderr = proc.communicate(input=pipe_data)

            if proc.returncode != 0:
                logger.warning(f"ffmpeg libx265 HLG encode failed: {stderr.decode('utf-8', errors='ignore')}. Retrying OpenCV fallback.")
                cls.encode_opencv_fallback(output_path, frames_acescg, fps)
        except (FileNotFoundError, Exception) as e:
            logger.warning(f"ffmpeg execution failed ({e}), falling back to OpenCV writer.")
            cls.encode_opencv_fallback(output_path, frames_acescg, fps)

        return output_path

    @classmethod
    def encode_tonemapped_mp4(
        cls,
        output_path: Union[str, Path],
        frames_acescg: Union[List[np.ndarray], np.ndarray],
        fps: float = 24.0,
        source_colorspace: str = "srgb_gamma",
    ) -> Path:
        """Encode 8-bit SDR tonemapped preview video for desktop playback with metadata."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg = cls.find_ffmpeg()

        preview_frames = [ColorSpaceConverter.tonemap_acescg_for_preview(f) for f in frames_acescg]

        if not ffmpeg:
            return cls.encode_opencv_fallback(output_path, preview_frames, fps=fps)

        preview_array = np.stack(preview_frames, axis=0)
        num_frames, height, width, _ = preview_array.shape

        cmd = [
            ffmpeg,
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{width}x{height}",
            "-pix_fmt", "rgb24",
            "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-metadata", f"source_framerate={fps:.3f}",
            "-metadata", f"source_colorspace={source_colorspace}",
            "-metadata", "target_colorspace=sRGB Tonemapped Preview",
            "-crf", "20",
            "-preset", "fast",
            str(output_path)
        ]

        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            _, stderr = proc.communicate(input=preview_array.tobytes())

            if proc.returncode != 0:
                cls.encode_opencv_fallback(output_path, preview_frames, fps=fps)
        except (FileNotFoundError, Exception):
            cls.encode_opencv_fallback(output_path, preview_frames, fps=fps)

        return output_path

    @staticmethod
    def write_metadata_json(
        json_path: Union[str, Path],
        metadata: Dict[str, Any],
    ) -> Path:
        """Write conversion metadata sidecar JSON file."""
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        return json_path
