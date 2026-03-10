import torch
import time
import numpy as np
from transformers import SegformerForSemanticSegmentation

MODEL_DIR = "./segformer_model"
RUNS = 200
IMG_SIZE = 512

print("Loading model...")
model = SegformerForSemanticSegmentation.from_pretrained(MODEL_DIR)
model.eval().cuda()

dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).cuda()

# Warmup
print("Warming up (50 runs)...")
with torch.no_grad():
    for _ in range(50):
        _ = model(pixel_values=dummy)

# Benchmark FP32
print(f"Benchmarking FP32 ({RUNS} runs)...")
torch.cuda.synchronize()
times = []
with torch.no_grad():
    for _ in range(RUNS):
        start = time.perf_counter()
        _ = model(pixel_values=dummy)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)

fp32_avg = np.mean(times)
fp32_std = np.std(times)

# Benchmark FP16
print(f"Benchmarking FP16 ({RUNS} runs)...")
model_fp16 = model.half()
dummy_fp16 = dummy.half()
torch.cuda.synchronize()
times_fp16 = []
with torch.no_grad():
    for _ in range(RUNS):
        start = time.perf_counter()
        _ = model_fp16(pixel_values=dummy_fp16)
        torch.cuda.synchronize()
        times_fp16.append((time.perf_counter() - start) * 1000)

fp16_avg = np.mean(times_fp16)
fp16_std = np.std(times_fp16)

print(f"""
╔══════════════════════════════════════════╗
║       PyTorch Baseline Results           ║
╠══════════════════════════════════════════╣
║ FP32 Latency  : {fp32_avg:6.2f} ms ± {fp32_std:.2f}      ║
║ FP32 FPS      : {1000/fp32_avg:6.1f} FPS               ║
║ FP16 Latency  : {fp16_avg:6.2f} ms ± {fp16_std:.2f}      ║
║ FP16 FPS      : {1000/fp16_avg:6.1f} FPS               ║
║ VRAM (FP32)   : {torch.cuda.memory_allocated()/1e6:6.1f} MB               ║
╚══════════════════════════════════════════╝
""")

# Save results for later comparison
import json
results = {
    "pytorch_fp32": {"latency_ms": fp32_avg, "fps": 1000/fp32_avg},
    "pytorch_fp16": {"latency_ms": fp16_avg, "fps": 1000/fp16_avg},
}
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)
print("✅ Results saved to results.json")
