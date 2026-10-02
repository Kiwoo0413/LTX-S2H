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
