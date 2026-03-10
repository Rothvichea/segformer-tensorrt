# SegFormer TensorRT Optimization — Real-Time Semantic Segmentation

> Optimizing a transformer-based semantic segmentation model (SegFormer-b0) for real-time inference on embedded GPU hardware using TensorRT — targeting autonomous driving and robotics perception stacks.

---

## 🎯 Motivation

Autonomous driving and robotics systems require perception models that run in **real-time** under strict hardware constraints. Production vehicles use embedded GPUs (NVIDIA Drive Orin, Jetson) with limited VRAM and power budgets.

This project benchmarks the full optimization pipeline for **SegFormer-b0** — a transformer-based semantic segmentation model pretrained on Cityscapes (19 urban classes) — across four inference backends, and documents a critical numerical precision finding encountered during the process.

---

## 🏗️ Pipeline

```
SegFormer-b0 (HuggingFace)
    ↓
[1] PyTorch FP32 baseline
    ↓
[2] PyTorch FP16 (naive)
    ↓
[3] ONNX Export → ONNX Runtime (GPU)
    ↓
[4] TensorRT FP16 engine
    ↓
[5] TensorRT FP32 engine  ✅ (production-safe)
```

---

## 📊 Benchmark Results

Tested on **NVIDIA RTX 3060 (6GB VRAM)** — Dell XPS 17, Ubuntu 22.04, CUDA 12.8, TensorRT 10.15

| Backend | Latency (ms) | FPS | Speedup vs Baseline | Output Quality |
|---|---|---|---|---|
| PyTorch FP32 | 12.33 ± 0.91 | 81 | 1.00x (baseline) | ✅ Correct |
| PyTorch FP16 | 15.39 ± 87.65 | 65 | 0.80x ❌ | ✅ Correct |
| ONNX Runtime GPU | 13.35 ± 1.18 | 75 | 0.92x | ✅ Correct |
| TensorRT FP16 | **3.00 ± 0.16** | **334** | **4.12x** | ⚠️ Overflow |
| TensorRT FP32 | **7.25 ± 0.30** | **138** | **1.70x** | ✅ Correct |

> **Best production choice: TensorRT FP32 — 1.7x speedup with correct output**

---

## 🔍 Key Findings

### 1. PyTorch FP16 is actually slower than FP32
Naive `.half()` casting in PyTorch introduced a massive standard deviation (±87ms vs ±0.91ms for FP32). This is caused by CUDA kernel warm-up instability with small transformer models — FP16 benefits don't materialize without proper engine-level optimization.

### 2. ONNX Runtime GPU barely matches PyTorch
For small models like SegFormer-b0 (3.7M parameters), the ONNX Runtime session overhead cancels out the GPU execution gains. ONNX Runtime shows stronger benefits on larger models (>100M parameters).

### 3. TensorRT FP16 caused numerical overflow — a critical production finding
TensorRT FP16 achieved **4.12x speedup** but produced completely wrong segmentation output. Numerical analysis revealed:

```
PyTorch  logits range : [-39.4,  +9.2]   ✅ normal
TensorRT FP16 range   : [-512.0, +512.0] ❌ clipped — FP16 overflow
Max difference        : 547.2             ❌ completely wrong
```

SegFormer's internal attention mechanism produces intermediate values that **exceed FP16 range** (`[-65504, +65504]` theoretical max, but precision degrades well before that). TensorRT silently clips these values, producing fast but incorrect results.

**This is a known risk in transformer optimization** — and exactly why automotive-grade perception systems require numerical validation, not just latency benchmarks.

### 4. TensorRT FP32 is the correct production target
Switching to FP32 engine: **1.70x speedup with numerically correct output**.

```
Max difference PT vs TRT FP32 : < 0.01   ✅ numerically equivalent
```

---

## 🚗 Visual Results

SegFormer-b0 TensorRT FP32 running on a real urban street scene:

![Segmentation Result](segmentation_result.png)

*Left: Input image — Center: Segmentation map (19 Cityscapes classes) — Right: Overlay*

Classes detected: road, sidewalk, building, person, car, vegetation, sky, traffic light, traffic sign

---

## 🛠️ Setup & Reproduction

### Requirements

```bash
# Option A — conda environment file (recommended)
conda env create -f environment.yml
conda activate segformer-trt

# Option B — manual setup
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

### Run

```bash
# 1. Download model
python download_model.py

# 2. PyTorch baseline benchmark
python benchmark_pytorch.py

# 3. Export to ONNX
python export_onnx.py

# 4. ONNX Runtime benchmark
python benchmark_onnx.py

# 5. Build TensorRT engines
python build_tensorrt.py        # FP16 (fast, overflow)
python build_tensorrt_fp32.py   # FP32 (production-safe)

# 6. TensorRT benchmark
python benchmark_tensorrt.py

# 7. Visual demo
python visualize.py

# 8. Run inference on a video
# Edit INPUT_VIDEO on line 11 of inference_video.py to point to your own video file
# Supports .mp4, .webm, .avi, .mov — any file OpenCV can read
python inference_video.py
# Output saved as: segmentation_output_final.mp4
```

---

## 📁 Project Structure

```
segformer-tensorrt/
├── environment.yml                  # Conda environment (recommended)
├── download_model.py                # Download SegFormer-b0 from HuggingFace
├── benchmark_pytorch.py             # PyTorch FP32 + FP16 baseline
├── export_onnx.py                   # ONNX export pipeline
├── benchmark_onnx.py                # ONNX Runtime GPU benchmark
├── build_tensorrt.py                # TensorRT engine builder
├── benchmark_tensorrt.py            # TensorRT benchmark + comparison
├── inference_video.py               # Video inference
├── inference_realtime.py            # Real-time webcam inference
├── inference_realtime_fusion.py     # SegFormer + YOLOv8 fusion
├── visualize.py                     # Visual segmentation demo
├── generate_report.py               # Generate benchmark report
├── results.json                     # All benchmark results
├── plot_*.png                       # Benchmark charts
├── segmentation_result.png          # Visual output sample
└── sample.png                       # Input sample image
    # Note: model files (.onnx, .engine, segformer_model/) are excluded
    # Run download_model.py → export_onnx.py → build_tensorrt.py to regenerate
```

---

## 🧠 Model

- **Architecture:** SegFormer-b0 (Mix Transformer encoder + lightweight MLP decoder)
- **Parameters:** 3.7M
- **Dataset:** Cityscapes (19 urban classes, 512×1024 resolution)
- **Source:** `nvidia/segformer-b0-finetuned-cityscapes-512-1024` (HuggingFace)

---

## 🔗 References

- [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203) — Xie et al., NeurIPS 2021
- [NVIDIA TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/)
- [Cityscapes Dataset](https://www.cityscapes-dataset.com/)

---

## 👤 Author

**Rothvichea CHEA** — Mechatronics / Robotics Engineer  
Perception · ROS2 · Deep Learning · Embedded Systems  
[Portfolio](https://rothvicheachea.netlify.app) · [LinkedIn](https://www.linkedin.com/in/chea-rothvichea-a96154227/)
