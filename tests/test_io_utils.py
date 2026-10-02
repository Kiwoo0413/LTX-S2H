"""
tests/test_io_utils.py
Synthetic unit tests for color space conversion, EXR IO, and video frame math.
Zero host dependencies, 100% CPU runnable in seconds.
"""

import tempfile
from pathlib import Path
import numpy as np
import pytest

from core.io_utils import (
    ColorSpace,
    ColorSpaceConverter,
    EXRSequenceIO,
)
from core.hdr_engine import LTXHDREngine


def test_srgb_gamma_linear_roundtrip():
    """Verify sRGB gamma to scene-linear and back maintains numerical fidelity."""
    original = np.linspace(0.0, 1.0, 100, dtype=np.float32).reshape(10, 10, 1)
    linear = ColorSpaceConverter.srgb_gamma_to_linear(original)
    recovered = ColorSpaceConverter.linear_to_srgb_gamma(linear)

    np.testing.assert_allclose(original, recovered, atol=1e-4)


def test_acescg_acescct_roundtrip():
    """Verify ACEScg linear to ACEScct logarithmic and back maintains fidelity."""
    # Test values from deep shadows (0.001) to super-highlights (20.0)
    values = np.array([0.001, 0.005, 0.01, 0.1, 0.18, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0], dtype=np.float32)
    acescg = np.tile(values.reshape(-1, 1, 1), (1, 1, 3))

    acescct = ColorSpaceConverter.acescg_to_acescct(acescg)
    recovered = ColorSpaceConverter.acescct_to_acescg(acescct)

    np.testing.assert_allclose(acescg, recovered, rtol=1e-3, atol=1e-4)


def test_hlg_conversion():
    """Verify ACEScg linear signal converts to valid Rec.2100 HLG signal [0.0, 1.0]."""
    test_scene_linear = np.array([[[0.0, 0.0, 0.0], [0.18, 0.18, 0.18], [1.0, 1.0, 1.0], [5.0, 5.0, 5.0]]], dtype=np.float32)
    hlg = ColorSpaceConverter.acescg_to_hlg_bt2020(test_scene_linear)

    assert hlg.shape == test_scene_linear.shape
    assert np.all(hlg >= 0.0)
    assert np.all(hlg <= 1.0)
    # 0.0 in linear must map to 0.0 in HLG
    assert np.isclose(hlg[0, 0, 0], 0.0, atol=1e-4)


def test_exr_write_and_read():
    """Verify writing and reading half-float 16-bit EXR frames."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        exr_path = Path(tmp_dir) / "synthetic_test.exr"

        # Synthetic HDR test pattern with super-white (> 1.0) values
        synthetic_hdr = np.zeros((64, 64, 3), dtype=np.float32)
        synthetic_hdr[:, :, 0] = np.linspace(0.0, 4.0, 64)[:, None] # Red goes up to 4.0
        synthetic_hdr[:, :, 1] = np.linspace(0.0, 2.0, 64)[None, :] # Green goes up to 2.0
        synthetic_hdr[:, :, 2] = 0.5

        success = EXRSequenceIO.write_frame(exr_path, synthetic_hdr)
        assert success, "EXR write_frame must return True"
        assert exr_path.exists()

        # Read back
        read_back = EXRSequenceIO.read_frame(exr_path)
        assert read_back.shape == (64, 64, 3)
        # Verify values > 1.0 were not clipped by 8-bit quantization
        assert np.max(read_back[:, :, 0]) > 3.5
        np.testing.assert_allclose(synthetic_hdr, read_back, rtol=5e-2, atol=1e-3)


def test_ltx_frame_count_adjustment():
    """Verify temporal frame adjustment enforces 8k + 1 constraint."""
    assert LTXHDREngine.adjust_frame_count_for_ltx(1) == 1
    assert LTXHDREngine.adjust_frame_count_for_ltx(9) == 9
    assert LTXHDREngine.adjust_frame_count_for_ltx(16) == 9   # 8*1 + 1
    assert LTXHDREngine.adjust_frame_count_for_ltx(17) == 17  # 8*2 + 1
    assert LTXHDREngine.adjust_frame_count_for_ltx(20) == 17
    assert LTXHDREngine.adjust_frame_count_for_ltx(25) == 25  # 8*3 + 1
    assert LTXHDREngine.adjust_frame_count_for_ltx(30) == 25
    assert LTXHDREngine.adjust_frame_count_for_ltx(33) == 33  # 8*4 + 1
