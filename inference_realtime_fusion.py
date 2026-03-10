import os
import sys
import warnings
import logging

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
import threading
from transformers import SegformerImageProcessor
from ultralytics import YOLO
from PIL import Image

sys.stderr = stderr_backup

# ── Config ─────────────────────────────────────────────────────────────────────
ENGINE_PATH  = "./segformer_fp32.engine"
MODEL_DIR    = "./segformer_model"
INPUT_VIDEO  = "./nightdriving.webm"
IMG_SIZE     = 512
WINDOW_W     = 1280
WINDOW_H     = 400
SEG_CONF_THR = 0.75

# ── Cityscapes ─────────────────────────────────────────────────────────────────
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

YOLO_TO_SEG = {0:11,1:18,2:13,3:17,5:15,7:14,9:6,11:7}
YOLO_BOX_COLORS = {
    0:(0,0,255),1:(119,11,32),2:(0,0,142),3:(0,0,230),
    5:(0,60,100),7:(0,0,70),9:(250,170,30),11:(220,220,0),
}

# ── Shared result between inference thread and main thread ─────────────────────
result_lock   = threading.Lock()
latest_result = {"display": None, "lat_seg": 0.0, "lat_yolo": 0.0, "skipped": 0}
stop_event    = threading.Event()
pause_event   = threading.Event()

# ── Load models ────────────────────────────────────────────────────────────────
print("Loading SegFormer TensorRT FP32 engine...")
TRT_LOGGER    = trt.Logger(trt.Logger.ERROR)
runtime       = trt.Runtime(TRT_LOGGER)
with open(ENGINE_PATH, "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())
context       = engine.create_execution_context()
context.set_input_shape("pixel_values", (1, 3, IMG_SIZE, IMG_SIZE))
out_shape     = tuple(context.get_tensor_shape("logits"))
output_tensor = torch.zeros(out_shape, dtype=torch.float32).cuda().contiguous()
print("✅ SegFormer loaded!")

print("Loading YOLOv8m...")
yolo = YOLO("yolov8m.pt")
yolo.to("cuda")
print("✅ YOLOv8m loaded!")

processor = SegformerImageProcessor.from_pretrained(MODEL_DIR)

# ── Inference helpers ──────────────────────────────────────────────────────────
def run_segformer(frame_bgr, H, W):
    # CLAHE — adaptive contrast enhancement for night/low-light
    lab   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l     = clahe.apply(l)
    lab   = cv2.merge([l, a, b])
    frame_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img       = Image.fromarray(frame_rgb)
    inputs    = processor(images=img, return_tensors="pt")
    pv        = inputs["pixel_values"].float().cuda().contiguous()

    context.set_tensor_address("pixel_values", pv.data_ptr())
    context.set_tensor_address("logits",        output_tensor.data_ptr())
    context.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()

    logits          = output_tensor.float().cpu()
    up              = torch.nn.functional.interpolate(
        logits, size=(H, W), mode="bilinear", align_corners=False)
    probs           = torch.softmax(up, dim=1)
    max_prob, seg_t = probs.max(dim=1)
    seg_map         = seg_t.squeeze().numpy().astype(np.uint8)
    max_prob        = max_prob.squeeze().numpy()

    seg_map[max_prob < SEG_CONF_THR] = 2
    seg_map[seg_map == 11] = 2
    seg_map[seg_map == 12] = 2

    r = frame_rgb[:,:,0].astype(np.int16)
    g = frame_rgb[:,:,1].astype(np.int16)
    b = frame_rgb[:,:,2].astype(np.int16)
    seg_map[(g-r>15)&(g-b>10)&np.isin(seg_map,[2,3,11,12])] = 8
    seg_map[(b-r>20)&(b-g>10)&(seg_map==2)]                  = 10
    return seg_map

def run_yolo(frame_bgr):
    lab   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l     = clahe.apply(l)
    lab   = cv2.merge([l, a, b])
    frame_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return yolo(frame_bgr, verbose=False, conf=0.35, iou=0.45,
                classes=list(YOLO_TO_SEG.keys()))[0]

def apply_yolo_masks(seg_map, res, H, W):
    out = seg_map.copy()
    if res.boxes is None: return out
    for box in res.boxes:
        cls_id = int(box.cls.item())
        if cls_id not in YOLO_TO_SEG: continue
        x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
        x1,y1 = max(0,x1), max(0,y1)
        x2,y2 = min(W,x2), min(H,y2)
        roi   = out[y1:y2, x1:x2]
        roi[np.isin(roi,[2,3,8,9,10,11,12])] = YOLO_TO_SEG[cls_id]
        out[y1:y2, x1:x2] = roi
    return out

def build_display(frame, seg_map, res, lat_seg, lat_yolo, H, W, skipped):
    mask_bgr = cv2.cvtColor(COLORS[seg_map], cv2.COLOR_RGB2BGR)
    overlay  = cv2.addWeighted(frame, 0.50, mask_bgr, 0.50, 0)

    if res.boxes is not None:
        for box in res.boxes:
            cls_id          = int(box.cls.item())
            conf            = float(box.conf.item())
            x1,y1,x2,y2    = map(int, box.xyxy[0].tolist())
            color           = YOLO_BOX_COLORS.get(cls_id, (255,255,255))
            label           = f"{res.names[cls_id]} {conf:.2f}"
            cv2.rectangle(overlay, (x1,y1), (x2,y2), color, 2)
            (tw,th),_       = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(overlay, (x1,y1-th-6), (x1+tw+4,y1), color, -1)
            cv2.putText(overlay, label, (x1+2,y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    combined = np.hstack([frame, overlay])
    total_ms = lat_seg + lat_yolo

    cv2.rectangle(combined, (0,0), (780,40), (20,20,20), -1)
    cv2.putText(combined,
        f"Seg {lat_seg:.0f}ms + YOLO {lat_yolo:.0f}ms = {total_ms:.0f}ms | "
        f"{1000/max(total_ms,1):.0f} FPS | skipped {skipped}",
        (10,27), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0,255,120), 2)

    cv2.putText(combined, "Original",
                (10,H-10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)
    cv2.putText(combined, "SegFormer + YOLOv8m Fusion",
                (W+10,H-10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)

    present = np.unique(seg_map)
    for i, cls_id in enumerate(present[:10]):
        c   = COLORS[cls_id]
        bgr = (int(c[2]),int(c[1]),int(c[0]))
        x   = 10 + i*120
        cv2.rectangle(combined, (x,H-22), (x+12,H-8), bgr, -1)
        cv2.putText(combined, CLASSES[cls_id],
                    (x+16,H-9), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255,255,255), 1)

    return cv2.resize(combined, (WINDOW_W, WINDOW_H))

# ── Inference thread ───────────────────────────────────────────────────────────
def inference_thread(cap, H, W):
    skipped = 0
    while not stop_event.is_set():
        if pause_event.is_set():
            time.sleep(0.05)
            continue

        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        # Normalize frame size for consistent latency
        frame = cv2.resize(frame, (640, 360))
        time.sleep(0.03)  # ~30 FPS cap — increase to slow down more

        # If a result is already waiting and not consumed, skip this frame
        with result_lock:
            has_pending = latest_result["display"] is not None
        if has_pending:
            skipped += 1
            continue

        try:
            t0      = time.perf_counter()
            seg_map = run_segformer(frame, H, W)
            lat_seg = (time.perf_counter() - t0) * 1000

            t1       = time.perf_counter()
            res      = run_yolo(frame)
            lat_yolo = (time.perf_counter() - t1) * 1000

            seg_map  = apply_yolo_masks(seg_map, res, H, W)
            display  = build_display(frame, seg_map, res,
                                     lat_seg, lat_yolo, H, W, skipped)

            with result_lock:
                latest_result["display"]  = display
                latest_result["lat_seg"]  = lat_seg
                latest_result["lat_yolo"] = lat_yolo
                latest_result["skipped"]  = skipped

        except Exception as e:
            print(f"Inference error: {e}")
            continue

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    cap    = cv2.VideoCapture(INPUT_VIDEO)
    W_in = 640
    H_in = 360
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video : {W_in}x{H_in} @ {fps_in:.1f} FPS — {total} frames")
    print("Controls: Q/ESC = quit  |  SPACE = pause  |  R = restart")

    # Start inference thread
    t = threading.Thread(target=inference_thread, args=(cap, H_in, W_in), daemon=True)
    t.start()

    WIN = "SegFormer + YOLOv8m — Real-Time Perception"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, WINDOW_W, WINDOW_H)

    # Show loading screen while inference warms up
    loading = np.zeros((WINDOW_H, WINDOW_W, 3), dtype=np.uint8)
    cv2.putText(loading, "Loading... warming up inference",
                (WINDOW_W//2 - 220, WINDOW_H//2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,120), 2)
    cv2.imshow(WIN, loading)
    cv2.waitKey(1)

    while not stop_event.is_set():
        # Grab latest result
        with result_lock:
            display = latest_result["display"]
            latest_result["display"] = None  # consume it

        if display is not None:
            cv2.imshow(WIN, display)

        key = cv2.waitKey(16) & 0xFF  # ~60fps display loop
        if key in [ord("q"), 27]:
            stop_event.set()
        elif key == ord(" "):
            if pause_event.is_set():
                pause_event.clear(); print("▶  Resumed")
            else:
                pause_event.set();   print("⏸  Paused")
        elif key == ord("r"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            with result_lock:
                latest_result["skipped"] = 0
                latest_result["display"] = None
            print("🔄 Restarted")

    stop_event.set()
    t.join(timeout=3)
    cap.release()
    cv2.destroyAllWindows()

    print(f"\n{'='*45}")
    print(f"Last Seg    : {latest_result['lat_seg']:.1f} ms")
    print(f"Last YOLO   : {latest_result['lat_yolo']:.1f} ms")
    print(f"Total       : {latest_result['lat_seg']+latest_result['lat_yolo']:.1f} ms")
    print(f"Skipped     : {latest_result['skipped']} frames")
    print(f"{'='*45}")

if __name__ == "__main__":
    main()