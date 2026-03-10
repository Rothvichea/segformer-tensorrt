import torch
from transformers import SegformerForSemanticSegmentation

MODEL_DIR = "./segformer_model"
ONNX_PATH = "./segformer.onnx"
IMG_SIZE = 512

print("Loading model...")
model = SegformerForSemanticSegmentation.from_pretrained(MODEL_DIR)
model.eval().cuda()

dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).cuda()

print("Exporting to ONNX...")
torch.onnx.export(
    model,
    (dummy,),
    ONNX_PATH,
    input_names=["pixel_values"],
    output_names=["logits"],
    dynamic_axes={
        "pixel_values": {0: "batch_size"},
        "logits": {0: "batch_size"}
    },
    opset_version=17,
    do_constant_folding=True,
)
print(f"✅ ONNX model saved to {ONNX_PATH}")

# Verify ONNX model
import onnx
model_onnx = onnx.load(ONNX_PATH)
onnx.checker.check_model(model_onnx)
print("✅ ONNX model verified — graph is valid")

# Check file size
import os
size_mb = os.path.getsize(ONNX_PATH) / 1e6
print(f"📦 ONNX file size: {size_mb:.1f} MB")
