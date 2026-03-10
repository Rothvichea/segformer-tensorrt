import os
import sys
import warnings
import logging

# Suppress all Qt/OpenCV warnings before anything else
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false"
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)
stderr_backup = sys.stderr
sys.stderr = open(os.devnull, "w")

import torch
import tensorrt as trt
import numpy as np
import cv2
import time
from transformers import SegformerImageProcessor
from PIL import Image

# Restore stderr after noisy imports
sys.stderr = stderr_backup

# ── Config ────────────────────────────────────────────────────────────────────
ENGINE_PATH = "./segformer_fp32.engine"
MODEL_DIR   = "./segformer_model"
INPUT_VIDEO = "./driving_city.mp4"
IMG_SIZE    = 512
WINDOW_W    = 1280   # display width  (side-by-side)
WINDOW_H    = 360    # display height

# ── Cityscapes classes + colors ───────────────────────────────────────────────
CLASSES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"
]
COLORS = np.array([
    (128,64,128),(244,35,232),(70,70,70),(102,102,156),(190,153,153),
    (153,153,153),(250,170,30),(220,220,0),(107,142,35),(152,251,152),
    (70,130,180),(220,20,60),(255,0,0),(0,0,142),(0,0,70),(0,60,100),
    (0,80,100),(0,0,230),(119,11,32)
], dtype=np.uint8)

# ── Load TensorRT engine ──────────────────────────────────────────────────────
print("Loading TensorRT FP32 engine...")
TRT_LOGGER = trt.Logger(trt.Logger.ERROR)   # ERROR only — no spam
runtime    = trt.Runtime(TRT_LOGGER)
with open(ENGINE_PATH, "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()
context.set_input_shape("pixel_values", (1, 3, IMG_SIZE, IMG_SIZE))
out_shape     = tuple(context.get_tensor_shape("logits"))
output_tensor = torch.zeros(out_shape, dtype=torch.float32).cuda().contiguous()
print("✅ Engine loaded!")

# ── Load processor ────────────────────────────────────────────────────────────
processor = SegformerImageProcessor.from_pretrained(MODEL_DIR)

# ── Inference helpers ─────────────────────────────────────────────────────────
def preprocess(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img       = Image.fromarray(frame_rgb)
    inputs    = processor(images=img, return_tensors="pt")
    return inputs["pixel_values"].float().cuda().contiguous()

def infer(pixel_values):
    context.set_tensor_address("pixel_values", pixel_values.data_ptr())
    context.set_tensor_address("logits",        output_tensor.data_ptr())
    context.execute_async_v3(
        stream_handle=torch.cuda.current_stream().cuda_stream
    )
    torch.cuda.synchronize()
    return output_tensor

def postprocess(logits, H, W):
    up = torch.nn.functional.interpolate(
        logits.float().cpu(), size=(H, W),
        mode="bilinear", align_corners=False
    )
    seg = up.argmax(dim=1).squeeze().numpy().astype(np.uint8)
    return seg, COLORS[seg]

def draw_overlay(frame, color_mask, seg_map, latency, avg_latency):
    H, W = frame.shape[:2]

    # Blend segmentation over frame
    mask_bgr = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)
    overlay  = cv2.addWeighted(frame, 0.55, mask_bgr, 0.45, 0)

    # Side-by-side
    combined = np.hstack([frame, overlay])

    # Top stats bar
    cv2.rectangle(combined, (0, 0), (560, 38), (0, 0, 0), -1)
    cv2.putText(
        combined,
        f"TensorRT FP32  |  {latency:.1f}ms  |  {1000/latency:.0f} FPS  |  avg {avg_latency:.1f}ms",
        (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 120), 2
    )

    # Bottom legend — present classes only
    present = np.unique(seg_map)
    for i, cls_id in enumerate(present[:10]):
        c   = COLORS[cls_id]
        bgr = (int(c[2]), int(c[1]), int(c[0]))
        x   = 10 + i * 118
        y   = H - 6
        cv2.rectangle(combined, (x, y - 14), (x + 12, y), bgr, -1)
        cv2.putText(combined, CLASSES[cls_id],
                    (x + 16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255,255,255), 1)

    # Resize to fixed display size
    return cv2.resize(combined, (WINDOW_W, WINDOW_H))

# ── Open video ────────────────────────────────────────────────────────────────
cap   = cv2.VideoCapture(INPUT_VIDEO)
W_in  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H_in  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_in = cap.get(cv2.CAP_PROP_FPS)
total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video : {W_in}x{H_in} @ {fps_in:.1f} FPS — {total} frames")
print("Controls: Q = quit  |  SPACE = pause/resume  |  R = restart")

WINDOW_NAME = "SegFormer TensorRT FP32 — Real-Time Segmentation"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, WINDOW_W, WINDOW_H)

latencies = []
paused    = False
frame_idx = 0

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            # Loop video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_idx = 0
            continue

        # Run pipeline
        t0                  = time.perf_counter()
        pixel_values        = preprocess(frame)
        logits              = infer(pixel_values)
        seg_map, color_mask = postprocess(logits, H_in, W_in)
        latency             = (time.perf_counter() - t0) * 1000
        latencies.append(latency)
        frame_idx += 1

        avg = np.mean(latencies[-30:]) if len(latencies) >= 30 else np.mean(latencies)
        display = draw_overlay(frame, color_mask, seg_map, latency, avg)
        cv2.imshow(WINDOW_NAME, display)

    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:       # Q or ESC
        break
    elif key == ord(" "):                   # SPACE — pause
        paused = not paused
        print("⏸  Paused" if paused else "▶  Resumed")
    elif key == ord("r"):                   # R — restart
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_idx = 0
        latencies.clear()
        print("🔄 Restarted")

cap.release()
cv2.destroyAllWindows()

if latencies:
    final_avg = np.mean(latencies[10:])
    print(f"\n{'='*40}")
    print(f"Frames processed : {frame_idx}")
    print(f"Avg latency      : {final_avg:.2f} ms")
    print(f"Avg FPS          : {1000/final_avg:.0f}")
    print(f"{'='*40}")