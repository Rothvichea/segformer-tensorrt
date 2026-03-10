import torch
import tensorrt as trt
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from transformers import SegformerImageProcessor
import time

ENGINE_PATH = "./segformer_fp32.engine"
IMAGE_PATH  = "./segformer_model/sdkkk.jpg"
MODEL_DIR   = "./segformer_model"

CLASSES = [
    "road","sidewalk","building","wall","fence","pole",
    "traffic light","traffic sign","vegetation","terrain","sky",
    "person","rider","car","truck","bus","train","motorcycle","bicycle"
]
COLORS = [
    (128,64,128),(244,35,232),(70,70,70),(102,102,156),(190,153,153),
    (153,153,153),(250,170,30),(220,220,0),(107,142,35),(152,251,152),
    (70,130,180),(220,20,60),(255,0,0),(0,0,142),(0,0,70),(0,60,100),
    (0,80,100),(0,0,230),(119,11,32)
]

# Load image + preprocess
img = Image.open(IMAGE_PATH).convert("RGB")
H, W = img.size[1], img.size[0]
processor = SegformerImageProcessor.from_pretrained(MODEL_DIR)
inputs = processor(images=img, return_tensors="pt")
pixel_values = inputs["pixel_values"].float().cuda().contiguous()

print(f"Input shape : {pixel_values.shape}")
print(f"Input range : [{pixel_values.min():.2f}, {pixel_values.max():.2f}]")

# Load TensorRT FP32 engine
print("Loading TensorRT FP32 engine...")
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(TRT_LOGGER)
with open(ENGINE_PATH, "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()
context.set_input_shape("pixel_values", (1, 3, 512, 512))
output_shape  = tuple(context.get_tensor_shape("logits"))
output_tensor = torch.zeros(output_shape, dtype=torch.float32).cuda().contiguous()

print(f"Output shape: {output_shape}")

def infer():
    context.set_tensor_address("pixel_values", pixel_values.data_ptr())
    context.set_tensor_address("logits", output_tensor.data_ptr())
    context.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()

# Warmup + benchmark
print("Warming up...")
for _ in range(10): infer()

times = []
for _ in range(100):
    start = time.perf_counter()
    infer()
    times.append((time.perf_counter() - start) * 1000)

avg_latency = np.mean(times[10:])
print(f"Logits range: [{output_tensor.min().item():.2f}, {output_tensor.max().item():.2f}]")
print(f"Steady latency: {avg_latency:.2f} ms | {1000/avg_latency:.0f} FPS")

# Post-process
logits = output_tensor.float().cpu()
upsampled = torch.nn.functional.interpolate(
    logits, size=(H, W), mode="bilinear", align_corners=False
)
seg_map = upsampled.argmax(dim=1).squeeze().numpy()

# Color mask
color_mask = np.zeros((H, W, 3), dtype=np.uint8)
for cls_id, color in enumerate(COLORS):
    color_mask[seg_map == cls_id] = color

# Overlay
img_np = np.array(img)
overlay = (img_np * 0.55 + color_mask * 0.45).astype(np.uint8)

present_classes = np.unique(seg_map)
print(f"Classes detected: {[CLASSES[i] for i in present_classes]}")

# Plot
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(
    f"SegFormer-b0 TensorRT FP32 — {avg_latency:.1f} ms | {1000/avg_latency:.0f} FPS",
    fontsize=13, fontweight="bold"
)
axes[0].imshow(img_np);     axes[0].set_title("Input Image");      axes[0].axis("off")
axes[1].imshow(color_mask); axes[1].set_title("Segmentation Map"); axes[1].axis("off")
axes[2].imshow(overlay);    axes[2].set_title("Overlay");          axes[2].axis("off")

patches = [mpatches.Patch(color=np.array(COLORS[i])/255, label=CLASSES[i])
           for i in present_classes]
fig.legend(handles=patches, loc="lower center", ncol=7, fontsize=9,
           bbox_to_anchor=(0.5, -0.02))

plt.tight_layout()
plt.savefig("segmentation_result.png", dpi=150, bbox_inches="tight")
print("✅ Saved segmentation_result.png")