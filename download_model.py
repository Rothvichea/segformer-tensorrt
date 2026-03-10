from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import torch

MODEL_NAME = "nvidia/segformer-b0-finetuned-cityscapes-512-1024"
SAVE_DIR = "./segformer_model"

print("Downloading SegFormer b0 (Cityscapes)...")
processor = SegformerImageProcessor.from_pretrained(MODEL_NAME)
model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME)

processor.save_pretrained(SAVE_DIR)
model.save_pretrained(SAVE_DIR)

print(f"✅ Model saved to {SAVE_DIR}")
print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

# Quick test
model.eval().cuda()
dummy = torch.randn(1, 3, 512, 512).cuda()
with torch.no_grad():
    out = model(pixel_values=dummy)
print(f"✅ Forward pass OK — output shape: {out.logits.shape}")
