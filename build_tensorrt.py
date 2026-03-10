import tensorrt as trt
import numpy as np
import os

ONNX_PATH = "./segformer.onnx"
ENGINE_FP16_PATH = "./segformer_fp16.engine"
IMG_SIZE = 512

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_path, engine_path, fp16=True):
    print(f"Building TensorRT engine (FP16={fp16})...")
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"ONNX parse error: {parser.get_error(i)}")
            return None

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)  # 2GB

    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("  FP16 mode enabled")

    # Optimization profile for dynamic batch
    profile = builder.create_optimization_profile()
    profile.set_shape("pixel_values",
        min=(1, 3, IMG_SIZE, IMG_SIZE),
        opt=(1, 3, IMG_SIZE, IMG_SIZE),
        max=(4, 3, IMG_SIZE, IMG_SIZE)
    )
    config.add_optimization_profile(profile)

    print("  Building engine (this takes 1-3 minutes)...")
    serialized = builder.build_serialized_network(network, config)

    if serialized is None:
        print("❌ Failed to build engine")
        return None

    with open(engine_path, "wb") as f:
        f.write(serialized)

    size_mb = os.path.getsize(engine_path) / 1e6
    print(f"✅ Engine saved to {engine_path} ({size_mb:.1f} MB)")
    return engine_path

# Build FP16 engine
build_engine(ONNX_PATH, ENGINE_FP16_PATH, fp16=True)
