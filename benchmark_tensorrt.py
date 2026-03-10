import tensorrt as trt
import numpy as np
import time
import json
import ctypes

ENGINE_PATH = "./segformer_fp16.engine"
RUNS = 200
IMG_SIZE = 512

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

print("Loading TensorRT engine...")
runtime = trt.Runtime(TRT_LOGGER)
with open(ENGINE_PATH, "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()
context.set_input_shape("pixel_values", (1, 3, IMG_SIZE, IMG_SIZE))

# Use torch for memory management — no pycuda needed
import torch
input_tensor  = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, dtype=torch.float16).cuda()
output_shape  = tuple(context.get_tensor_shape("logits"))
output_tensor = torch.zeros(output_shape, dtype=torch.float16).cuda()

def infer():
    context.set_tensor_address("pixel_values", input_tensor.data_ptr())
    context.set_tensor_address("logits", output_tensor.data_ptr())
    context.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()

# Warmup
print("Warming up (50 runs)...")
for _ in range(50):
    infer()

# Benchmark
print(f"Benchmarking TensorRT FP16 ({RUNS} runs)...")
times = []
for _ in range(RUNS):
    start = time.perf_counter()
    infer()
    times.append((time.perf_counter() - start) * 1000)

avg = np.mean(times)
std = np.std(times)

with open("results.json") as f:
    results = json.load(f)

pt_latency = results["pytorch_fp32"]["latency_ms"]
speedup = pt_latency / avg

print(f"""
╔══════════════════════════════════════════╗
║       TensorRT FP16 Results              ║
╠══════════════════════════════════════════╣
║ Latency       : {avg:6.2f} ms ± {std:.2f}          ║
║ FPS           : {1000/avg:6.1f} FPS                  ║
║ Speedup vs PT : {speedup:6.2f}x                      ║
╚══════════════════════════════════════════╝

📊 Full Comparison:
  PyTorch  FP32 : {pt_latency:.2f} ms → {1000/pt_latency:.1f} FPS
  ONNX Runtime  : {results['onnx_runtime']['latency_ms']:.2f} ms → {results['onnx_runtime']['fps']:.1f} FPS
  TensorRT FP16 : {avg:.2f} ms → {1000/avg:.1f} FPS  🏆
""")

results["tensorrt_fp16"] = {"latency_ms": avg, "fps": 1000/avg, "speedup": speedup}
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)
print("✅ Results saved to results.json")