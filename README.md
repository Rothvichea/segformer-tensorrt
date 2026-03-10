# ⚡ SegFormer TensorRT — Real-Time Semantic Segmentation

<div align="center">

**Full optimization pipeline: HuggingFace weights → ONNX → TensorRT engine — with a critical FP16 numerical overflow discovery and a real-time SegFormer + YOLOv8m fusion system.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://python.org)
[![TensorRT](https://img.shields.io/badge/TensorRT-10.x-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/tensorrt)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-SegFormer--b0-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/nvidia/segformer-b0-finetuned-cityscapes-512-1024)
[![ONNX](https://img.shields.io/badge/ONNX-Runtime-005CED)](https://onnxruntime.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[**Results**](#-benchmark-results) · [**Key Findings**](#-key-findings) · [**Fusion Pipeline**](#-real-time-fusion-segformer--yolov8m) · [**Quick Start**](#️-setup--reproduction)

</div>

---

<div align="center">

![Segmentation Result](segmentation_result.png)
*SegFormer-b0 TensorRT FP32 — 19-class Cityscapes semantic segmentation at **138 FPS** on RTX 3060*

</div>

---

## 🎯 What This Is

Autonomous driving and robotics systems need perception models that run **in real-time** under strict hardware constraints — NVIDIA Jetson AGX Orin, Drive Orin, edge GPUs with limited VRAM and power budgets.

This project documents the **complete optimization pipeline** for **SegFormer-b0** (transformer-based semantic segmentation, pretrained on Cityscapes, 19 urban classes) across 5 inference backends, and uncovers a **critical numerical precision failure** in TensorRT FP16 that would silently corrupt a production self-driving perception stack.

> ⚠️ **The main finding:** TensorRT FP16 gives a 4.12× speedup — but produces completely wrong segmentation output due to attention layer overflow. TensorRT FP32 is the correct production target: 1.70× speedup with numerically equivalent output.

---

## 📊 Benchmark Results

**Hardware:** NVIDIA RTX 3060 (6GB VRAM) · Ubuntu 22.04 · CUDA 12.8 · TensorRT 10.x · 200 runs, batch 1

| Backend | Latency (ms) | Std (ms) | FPS | Speedup | Output |
|---------|-------------|----------|-----|---------|--------|
| PyTorch FP32 | 12.33 | ±0.91 | 81 | 1.00× baseline | ✅ Correct |
| PyTorch FP16 | 15.39 | ±87.65 ⚠️ | 65 | 0.80× | ✅ Correct |
| ONNX Runtime GPU | 13.35 | ±1.18 | 75 | 0.92× | ✅ Correct |
| TensorRT FP16 | 3.00 | ±0.16 | 334 | 4.12× | ❌ **Overflow** |
| **TensorRT FP32 ★** | **7.25** | **±0.30** | **138** | **1.70×** | **✅ Correct** |

★ **Production target** — 1.70× speedup, numerically equivalent to PyTorch (max diff < 0.01)

---

## 🔍 Key Findings

### ❌ Finding 1 — TensorRT FP16: Fast but Wrong

TensorRT FP16 is **4.12× faster** than baseline but produces completely broken segmentation.

Root cause: SegFormer's Mix Transformer (MiT) encoder generates intermediate attention activations that **exceed FP16 precision range**. TensorRT silently clips these values:

```
PyTorch FP32 logits  : [-39.5,  +9.2]     ✅ normal distribution
TensorRT FP16 logits : [-512.0, +512.0]    ❌ clipped — FP16 overflow
Max difference       :  547.2              ❌ completely wrong output
```

**Production lesson:** Latency benchmarks alone are not enough. **Numerical validation is mandatory** for any safety-critical perception system. This is exactly the kind of silent failure that would pass a speed test and fail in a real vehicle.

---

### ⚠️ Finding 2 — PyTorch FP16 is Slower than FP32

Naive `.half()` casting introduced catastrophic latency instability: **std ±87.65 ms** vs ±0.91 ms for FP32. CUDA kernel warm-up issues with small transformers mean FP16 benefits don't materialize without engine-level optimization (TensorRT or torch.compile).

---

### ✅ Finding 3 — TensorRT FP32 is the Production Target

```
TensorRT FP32 speedup : 1.70×  (12.33 ms → 7.25 ms)
Std deviation         : ±0.30 ms  (3× more stable than PyTorch FP32)
Max output difference : < 0.01    ✅ numerically equivalent
```

Stable, fast, correct. Ready for Jetson AGX Orin and Drive Orin deployment.

---

## 🚗 Real-Time Fusion — SegFormer + YOLOv8m

Beyond the benchmark, this repo includes a **production-grade real-time perception system** fusing scene-level segmentation with object-level detection — targeting autonomous driving and mobile robotics.

### Architecture

```
Video Frame
    ↓
CLAHE preprocessing (night vision enhancement)
    ↓
┌─────────────────────┐    ┌──────────────────┐
│ SegFormer TRT FP32  │    │   YOLOv8m        │
│ 7.25 ms — 19 classes│    │ ~15 ms — objects │
└──────────┬──────────┘    └────────┬─────────┘
           │                        │
           └──────── Fusion ────────┘
                       ↓
         YOLOv8 boxes correct SegFormer masks
         for dynamic objects (person, car, truck...)
                       ↓
              Display overlay @ ~45 FPS
```

### Threading Design
- **Inference thread** — runs SegFormer + YOLO, writes to shared buffer with mutex lock
- **Display thread** — renders at 60 Hz, always responsive, frame-skipping for real-time sync

### CLAHE Night Vision
Contrast Limited Adaptive Histogram Equalization in LAB color space applied before inference — significantly improves person and road detection on night driving footage without overamplifying streetlights.

### Why Fusion?
SegFormer-b0 is a small model (3.7M params). On dynamic objects (people, vehicles) it tends to misclassify at boundaries. YOLOv8m bounding boxes override the segmentation mask for detected dynamic objects — combining the strengths of both models.

---

## 🏗️ Full Optimization Pipeline

```
SegFormer-b0 (HuggingFace)
    ↓
[1] PyTorch FP32 baseline         → 12.33 ms, 81 FPS
    ↓
[2] PyTorch FP16 naive            → 15.39 ms, 65 FPS ❌ unstable
    ↓
[3] ONNX export → Runtime GPU     → 13.35 ms, 75 FPS
    ↓
[4] TensorRT FP16 engine          → 3.00 ms, 334 FPS ❌ overflow
    ↓
[5] TensorRT FP32 engine ✅       → 7.25 ms, 138 FPS — production safe
```

---

## 🛠️ Setup & Reproduction

### Requirements
```bash
# Recommended — conda environment
conda env create -f environment.yml
conda activate segformer-trt

# Manual setup
conda create -n segformer-trt python=3.10 -y
conda activate segformer-trt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers ultralytics opencv-python Pillow matplotlib
pip install onnx onnxruntime-gpu tensorrt pycuda
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12
```

### LD_LIBRARY_PATH (required for ONNX Runtime GPU)
```bash
export LD_LIBRARY_PATH=/home/$USER/.local/lib/python3.10/site-packages/nvidia/cublas/lib:\
/home/$USER/.local/lib/python3.10/site-packages/nvidia/cudnn/lib:\
/home/$USER/.local/lib/python3.10/site-packages/nvidia/cufft/cu12/lib:$LD_LIBRARY_PATH
```

### Run Full Pipeline
```bash
# 1. Download model from HuggingFace
python download_model.py

# 2. Baseline benchmarks
python benchmark_pytorch.py

# 3. Export to ONNX + benchmark
python export_onnx.py
python benchmark_onnx.py

# 4. Build TensorRT engines
python build_tensorrt.py          # FP16 (demonstrates overflow)
python build_tensorrt_fp32.py     # FP32 (production-safe)

# 5. TensorRT benchmark + comparison
python benchmark_tensorrt.py

# 6. Visual demo
python visualize.py

# 7. Run on video (edit INPUT_VIDEO path in inference_video.py)
python inference_video.py
# Output: segmentation_output_final.mp4

# 8. Real-time fusion (SegFormer + YOLOv8m)
python inference_realtime_fusion.py
```

---

## 📁 Project Structure

```
segformer-tensorrt/
├── environment.yml                   # Conda environment
├── download_model.py                 # Download SegFormer-b0 from HuggingFace
├── benchmark_pytorch.py              # PyTorch FP32 + FP16 baseline
├── export_onnx.py                    # ONNX export pipeline
├── benchmark_onnx.py                 # ONNX Runtime GPU benchmark
├── build_tensorrt.py                 # TensorRT FP16 engine builder
├── build_tensorrt_fp32.py            # TensorRT FP32 engine builder
├── benchmark_tensorrt.py             # Full benchmark + comparison table
├── inference_video.py                # Video inference (any format)
├── inference_realtime.py             # Real-time webcam inference
├── inference_realtime_fusion.py      # SegFormer + YOLOv8m fusion
├── visualize.py                      # Visual segmentation demo
├── generate_report.py                # Generate markdown report
├── results.json                      # All benchmark results (raw)
├── plot_benchmark.png                # Latency + FPS comparison chart
├── plot_speedup.png                  # Speedup waterfall chart
├── plot_precision.png                # FP16 overflow analysis
├── plot_pipeline.png                 # Fusion pipeline diagram
├── segmentation_result.png           # Visual output sample
└── sample.png                        # Input sample image
    # Note: model files (.onnx, .engine, segformer_model/) excluded
    # Regenerate: download_model.py → export_onnx.py → build_tensorrt.py
```

---

## 🧠 Model Details

| Property | Value |
|----------|-------|
| Architecture | SegFormer-b0 (MiT encoder + MLP decoder) |
| Parameters | 3.7M |
| Dataset | Cityscapes (19 urban classes) |
| Input resolution | 512×1024 |
| Source | `nvidia/segformer-b0-finetuned-cityscapes-512-1024` |

---

## 📈 Benchmark Charts

| Latency & FPS | Speedup | FP16 Overflow |
|:---:|:---:|:---:|
| ![](plot_benchmark.png) | ![](plot_speedup.png) | ![](plot_precision.png) |

---

## 🔗 References

- [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203) — Xie et al., NeurIPS 2021
- [NVIDIA TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/)
- [Cityscapes Dataset](https://www.cityscapes-dataset.com/)
- [YOLOv8 — Ultralytics](https://github.com/ultralytics/ultralytics)

---

## 👤 Author

**Rothvichea CHEA** — Mechatronics Engineer | Robotics · Perception · Embedded AI

[![Portfolio](https://img.shields.io/badge/Portfolio-rothvicheachea.netlify.app-blue)](https://rothvicheachea.netlify.app)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?logo=linkedin)](https://www.linkedin.com/in/chea-rothvichea-a96154227/)
[![Email](https://img.shields.io/badge/Email-chearothvichea0599@gmail.com-red?logo=gmail)](mailto:chearothvichea0599@gmail.com)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
