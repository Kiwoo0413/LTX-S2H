"""
tests/test_nodes.py
Unit tests for LTXSDRToHDRNode instantiation, parameter schema, and defaults.
"""

from nodes.ltx_sdr_to_hdr_node import LTXSDRToHDRNode


def test_node_instantiation():
    """Verify LTXSDRToHDRNode instantiates cleanly outside Griptape."""
    node = LTXSDRToHDRNode()
    assert node is not None

    # Check input parameters
    assert "input_video" in node.parameters
    assert "input_colorspace" in node.parameters
    assert "keyframe_strength" in node.parameters
    assert "quantization" in node.parameters
    assert "inference_steps" in node.parameters
    assert "export_exr" in node.parameters
    assert "export_hlg" in node.parameters

    # Check output parameters
    assert "exr_sequence_dir" in node.parameters
    assert "hlg_video_path" in node.parameters
    assert "preview_video_path" in node.parameters
    assert "metadata_json_path" in node.parameters
    assert "framerate" in node.parameters
    assert "source_colorspace" in node.parameters
    assert "target_colorspace" in node.parameters
    assert "status" in node.parameters


def test_node_parameter_defaults():
    """Verify parameters have correct defaults."""
    node = LTXSDRToHDRNode()
    assert node.get_parameter_value("input_colorspace") == "srgb_gamma"
    assert node.get_parameter_value("keyframe_strength") == 0.95
    assert node.get_parameter_value("quantization") == "fp8-cast"
    assert node.get_parameter_value("inference_steps") == 8
    assert node.get_parameter_value("export_exr") is True
    assert node.get_parameter_value("export_hlg") is True


def test_node_parameter_mutation():
    """Verify parameter setters and getters work."""
    node = LTXSDRToHDRNode()
    node.set_parameter_value("input_video", "D:/test/video.mp4")
    assert node.get_parameter_value("input_video") == "D:/test/video.mp4"

    node.set_parameter_value("keyframe_strength", 0.90)
    assert node.get_parameter_value("keyframe_strength") == 0.90


def test_node_dynamic_output_path_resolution(tmp_path):
    """Verify that node.process() immediately resolves and sets output paths and metadata dynamically."""
    import cv2
    import numpy as np

    fake_video = tmp_path / "test_clip.mp4"
    # Create valid 10-frame dummy video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(fake_video), fourcc, 24.0, (64, 64))
    for _ in range(10):
        writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
    writer.release()

    node = LTXSDRToHDRNode()
    node.set_parameter_value("input_video", str(fake_video))
    node.process()

    expected_out_dir = tmp_path / "test_clip_HDR"
    assert node.get_parameter_value("output_dir") == str(expected_out_dir)
    assert node.get_parameter_value("exr_sequence_dir") == str(expected_out_dir / "acescg_exr")
    assert node.get_parameter_value("hlg_video_path") == str(expected_out_dir / "test_clip_HLG.mp4")
    assert node.get_parameter_value("preview_video_path") == str(expected_out_dir / "test_clip_HDR_preview.mp4")
    assert node.get_parameter_value("frame_count") > 0
    assert node.get_parameter_value("framerate") == 24.0
    assert node.get_parameter_value("source_colorspace") == "srgb_gamma"
    assert "ACEScg" in node.get_parameter_value("target_colorspace")
    assert "test_clip_metadata.json" in node.get_parameter_value("metadata_json_path")
    assert "Generated" in node.get_parameter_value("status") or "Successfully" in node.get_parameter_value("status")
