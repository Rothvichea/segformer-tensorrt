import torch
import tensorrt as trt
import numpy as np
import cv2
import time
from transformers import SegformerImageProcessor
from PIL import Image

ENGINE_PATH = "./segformer_fp32.engine"
MODEL_DIR   = "./segformer_model"
INPUT_VIDEO = "./driving.mp4"
OUTPUT_WEBM = "./segmentation_output.webm"
OUTPUT_MP4  = "./segmentation_output.mp4"
IMG_SIZE    = 512

CLASSES = [
    "road","sidewalk","building","wall","fence","pole",
    "traffic light","traffic sign","vegetation","terrain","sky",
    "person","rider","car","truck","bus","train","motorcycle","bicycle"
]
COLORS = np.array([
    (128,64,128),(244,35,232),(70,70,70),(102,102,156),(190,153,153),
    (153,153,153),(250,170,30),(220,220,0),(107,142,35),(152,251,152),
    (70,130,180),(220,20,60),(255,0,0),(0,0,142),(0,0,70),(0,60,100),
    (0,80,100),(0,0,230),(119,11,32)
], dtype=np.uint8)

# Load TensorRT engine
print("Loading TensorRT FP32 engine...")
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(TRT_LOGGER)
with open(ENGINE_PATH, "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()
context.set_input_shape("pixel_values", (1, 3, IMG_SIZE, IMG_SIZE))
out_shape     = tuple(context.get_tensor_shape("logits"))
output_tensor = torch.zeros(out_shape, dtype=torch.float32).cuda().contiguous()
print(f"✅ Engine loaded — output shape: {out_shape}")

# Load processor
processor = SegformerImageProcessor.from_pretrained(MODEL_DIR)

def preprocess(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    inputs = processor(images=img, return_tensors="pt")
    return inputs["pixel_values"].float().cuda().contiguous()

def infer(pixel_values):
    context.set_tensor_address("pixel_values", pixel_values.data_ptr())
    context.set_tensor_address("logits", output_tensor.data_ptr())
    context.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    return output_tensor

def postprocess(logits, H, W):
    upsampled = torch.nn.functional.interpolate(
        logits.float().cpu(), size=(H, W), mode="bilinear", align_corners=False
    )
    seg_map = upsampled.argmax(dim=1).squeeze().numpy().astype(np.uint8)
    color_mask = COLORS[seg_map]  # fast vectorized lookup
    return seg_map, color_mask

def draw_frame(frame_bgr, color_mask, seg_map, latency):
    H, W = frame_bgr.shape[:2]
    # Overlay
    color_mask_bgr = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(frame_bgr, 0.55, color_mask_bgr, 0.45, 0)

    # Side by side: original | overlay
    combined = np.hstack([frame_bgr, overlay])

    # Stats bar
    fps = 1000 / latency
    cv2.rectangle(combined, (0, 0), (420, 36), (0, 0, 0), -1)
    cv2.putText(combined, f"TensorRT FP32  |  {latency:.1f} ms  |  {fps:.0f} FPS",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 120), 2)

    # Mini legend — show top 6 present classes
    present = np.unique(seg_map)[:6]
    for i, cls_id in enumerate(present):
        color = tuple(int(c) for c in COLORS[cls_id])
        bgr   = (color[2], color[1], color[0])
        x = 8 + i * 120
        cv2.rectangle(combined, (x, H+4), (x+14, H+18), bgr, -1)
        cv2.putText(combined, CLASSES[cls_id], (x+18, H+16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255,255,255), 1)

    return combined

# Open video
cap = cv2.VideoCapture(INPUT_VIDEO)
fps_in  = cap.get(cv2.CAP_PROP_FPS)
W_in    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H_in    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
W_out   = W_in * 2  # side by side

print(f"Input  : {W_in}x{H_in} @ {fps_in:.1f} FPS — {total} frames")
print(f"Output : {W_out}x{H_in} side-by-side")

# Writers — WEBM + MP4
fourcc_webm = cv2.VideoWriter_fourcc(*"VP80")
fourcc_mp4  = cv2.VideoWriter_fourcc(*"mp4v")
writer_webm = cv2.VideoWriter(OUTPUT_WEBM, fourcc_webm, fps_in, (W_out, H_in))
writer_mp4  = cv2.VideoWriter(OUTPUT_MP4,  fourcc_mp4,  fps_in, (W_out, H_in))

# Process frames
latencies = []
frame_idx = 0
print("Processing frames...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    t0 = time.perf_counter()
    pixel_values = preprocess(frame)
    logits       = infer(pixel_values)
    seg_map, color_mask = postprocess(logits, H_in, W_in)
    latency = (time.perf_counter() - t0) * 1000
    latencies.append(latency)

    out_frame = draw_frame(frame, color_mask, seg_map, latency)
    writer_webm.write(out_frame)
    writer_mp4.write(out_frame)

    frame_idx += 1
    if frame_idx % 30 == 0:
        print(f"  Frame {frame_idx}/{total} — {latency:.1f} ms | {1000/latency:.0f} FPS")

cap.release()
writer_webm.release()
writer_mp4.release()

avg = np.mean(latencies[10:])
print(f"""
✅ Done!
   Frames processed : {frame_idx}
   Avg latency      : {avg:.2f} ms
   Avg FPS          : {1000/avg:.0f}
   Output WEBM      : {OUTPUT_WEBM}
   Output MP4       : {OUTPUT_MP4}
""")

# Re-encode MP4 properly with ffmpeg for web compatibility
import subprocess
subprocess.run([
    "ffmpeg", "-y", "-i", OUTPUT_MP4,
    "-vcodec", "libx264", "-crf", "23", "-preset", "fast",
    "./segmentation_output_final.mp4"
], check=True)
print("✅ Final MP4 re-encoded with H.264 → segmentation_output_final.mp4")