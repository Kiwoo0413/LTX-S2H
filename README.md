# LTX SDR-To-HDR Library for Griptape Nodes

VFX-grade Image-Conditioned LoRA (IC-LoRA) High Dynamic Range (HDR) Video Conversion Toolkit for **Griptape Nodes Desktop**, based on Lightricks' foundation model [Lightricks/LTX-2.5-22b-IC-LoRA-SDR-To-HDR](https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-SDR-To-HDR).

---

## 🌟 Key Features

1. **Lightricks 22B Distilled Architecture**:
   - Uses frozen full-clip SDR reference conditioning at strength `1.0`.
   - Seam keyframe guides held at `0.95` strength for temporal cohesion and highlight recovery.
   - Fast 8-step distilled Euler denoising schedule (zero CFG, zero spatial guidance).
   - Dedicated precomputed scene embedding (no text encoder required at runtime).
2. **Dual VFX & Display Deliverables**:
   - **Scene-Linear ACEScg 16-bit Float OpenEXR Sequence**: Fully compatible with Foundry Nuke, DaVinci Resolve, and ACES 1.3 finishing pipelines.
   - **10-bit Rec.2100 HLG BT.2020 HEVC MP4 Master**: Direct playback on HDR TVs, monitors, and Apple HDR displays.
   - **Tonemapped Preview MP4**: Instant desktop verification on standard sRGB displays.
3. **Hardware Optimized for 16GB GPUs (RTX 4080)**:
   - On-the-fly FP8 downcasting (`--quantization fp8-cast`).
   - Seamless sequential CPU offload and latent VAE auto-tiling.
4. **Decoupled Dual-Layer Architecture**:
   - `core/`: 100% headless, zero host dependencies, pure algorithmic math & PyTorch inference.
   - `nodes/`: Griptape `DataNode` integration with robust fallback shims.

---

## 📁 Repository & Library Structure

```text
LTX_SDR_TO_HDR/
├── core/
│   ├── __init__.py
│   ├── hdr_engine.py         # LTX-2.5 IC-LoRA pipeline & Euler denoising
│   ├── io_utils.py           # ACEScct/ACEScg transforms, OpenEXR half-float, HLG 10-bit MP4
│   ├── model_manager.py      # HuggingFace cache scanner & path resolution
│   └── worker.py             # Headless CLI worker for CUDA acceleration
├── nodes/
│   ├── __init__.py
│   ├── griptape_compat.py    # Fallback compatibility shim for testing
│   └── ltx_sdr_to_hdr_node.py# Griptape DataNode (LTXSDRToHDRNode)
├── scripts/
│   └── download_models.py    # Status verification and model download helper
├── tests/
│   ├── conftest.py
│   ├── test_io_utils.py      # Synthetic color space & EXR unit tests
│   ├── test_model_manager.py # Model discovery tests
│   └── test_nodes.py         # Griptape node schema tests
├── griptape_nodes_library.json# Griptape library manifest
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start in Griptape Nodes Desktop

1. **Open Workflow**:
   In Griptape Nodes Desktop, open:

   ```text
   D:\AI\GripTape\sdrtohdr.py
   ```

2. **Connect & Configure**:
   - **Input SDR Video (`LoadVideo`)**: Select your SDR `.mp4`, `.mov`, or ProRes footage.
   - **LTX-2.5 SDR to HDR Converter (`LTXSDRToHDRNode`)**:
     - `Input Colorspace`: `srgb_gamma` (Standard SDR video)
     - `Keyframe Guide Strength`: `0.95` (Recommended)
     - `VRAM Quantization`: `fp8-cast` (Optimized for RTX 4080 16GB)
     - `Inference Steps`: `8`
     - `Export ACEScg EXR Sequence`: `True`
     - `Export 10-bit HLG MP4`: `True`
3. **Execute**:
   Run the workflow. The node outputs:
   - `exr_sequence_dir`: Path to the 16-bit ACEScg OpenEXR folder (with embedded `framesPerSecond`, `source_colorspace`, and `ColorSpace` header metadata).
   - `hlg_video_path`: Path to the 10-bit Rec.2100 HLG MP4 master (with VUI BT.2020 tags and container metadata).
   - `preview_video_path`: Path to the tonemapped SDR preview video.
   - `metadata_json_path`: Path to `<video_stem>_metadata.json` sidecar containing full framerate, colorspace, resolution, and pipeline parameters.
   - `framerate`: Original source video frame rate (fps).
   - `source_colorspace`: Identified source video color space.
   - `target_colorspace`: Deliverable HDR color spaces (`ACEScg / Rec.2100 HLG`).

---

## 📦 Model Weights Setup

The IC-LoRA adapter is automatically discovered from your local Hugging Face cache:

- `ltx-2.5-22b-ic-lora-sdr-to-hdr-1.0.safetensors`
- `ltx-2.5-22b-ic-lora-sdr-to-hdr-scene-emb.safetensors`

To check model status or download missing base LTX-2.5 components:

```powershell
python scripts/download_models.py --status
```

To download the base transformer and VAE from [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5):

```powershell
python scripts/download_models.py --download-base
```

---

## 🧪 Running Unit Tests

Run the synthetic, CPU-runnable test suite (takes ~1 second):

```powershell
pytest tests/
```
