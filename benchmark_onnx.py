import onnxruntime as ort
import numpy as np
import time
import json

ONNX_PATH = "./segformer.onnx"
RUNS = 200
IMG_SIZE = 512

# Load existing results
with open("results.json") as f:
    results = json.load(f)

# Setup ONNX Runtime with GPU
print("Loading ONNX model...")
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
session = ort.InferenceSession(ONNX_PATH, providers=providers)

# Check which provider is active
active_provider = session.get_providers()[0]
print(f"Active provider: {active_provider}")

dummy = np.random.randn(1, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)
input_name = session.get_inputs()[0].name

# Warmup
print("Warming up (50 runs)...")
for _ in range(50):
    session.run(None, {input_name: dummy})

# Benchmark
print(f"Benchmarking ONNX Runtime ({RUNS} runs)...")
times = []
for _ in range(RUNS):
    start = time.perf_counter()
    session.run(None, {input_name: dummy})
    times.append((time.perf_counter() - start) * 1000)

avg = np.mean(times)
std = np.std(times)
speedup = results["pytorch_fp32"]["latency_ms"] / avg

print(f"""
╔══════════════════════════════════════════╗
║       ONNX Runtime Results               ║
╠══════════════════════════════════════════╣
║ Provider      : {active_provider[:24]:24s} ║
║ Latency       : {avg:6.2f} ms ± {std:.2f}          ║
║ FPS           : {1000/avg:6.1f} FPS                  ║
║ Speedup vs PT : {speedup:6.2f}x                      ║
╚══════════════════════════════════════════╝
""")

results["onnx_runtime"] = {"latency_ms": avg, "fps": 1000/avg, "speedup": speedup}
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)
print("✅ Results saved to results.json")
