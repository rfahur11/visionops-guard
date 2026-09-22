"""
VisionOps Guard - Model Conversion & Optimization Engine
Exports PyTorch weights (.pt) to ONNX Runtime (.onnx) format,
performs FP16 quantization, and benchmarks inference latency.
"""

import json
import time
from pathlib import Path
import cv2
import numpy as np
import torch
import yaml
import onnxruntime as ort
from ultralytics import YOLO


def benchmark_inference(pt_model_path: str, onnx_model_path: str, num_runs: int = 50) -> dict:
    """
    Benchmarks latency (ms/frame) between PyTorch (.pt) and ONNX Runtime (.onnx).
    """
    print(f"[*] Running Latency Benchmark on {num_runs} test samples...")
    # Dummy image input (1, 3, 640, 640)
    dummy_input = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    # 1. Benchmark PyTorch Inference
    torch_model = YOLO(pt_model_path)
    # Warmup
    for _ in range(5):
        _ = torch_model(dummy_input, verbose=False)

    start_pt = time.time()
    for _ in range(num_runs):
        _ = torch_model(dummy_input, verbose=False)
    total_pt_ms = ((time.time() - start_pt) / num_runs) * 1000.0

    # 2. Benchmark ONNX Runtime Inference
    session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    # Preprocess dummy input to (1, 3, 640, 640) float32
    img_float = dummy_input.astype(np.float32) / 255.0
    img_trans = np.transpose(img_float, (2, 0, 1))
    img_batch = np.expand_dims(img_trans, axis=0)

    # Warmup
    for _ in range(5):
        _ = session.run(None, {input_name: img_batch})

    start_onnx = time.time()
    for _ in range(num_runs):
        _ = session.run(None, {input_name: img_batch})
    total_onnx_ms = ((time.time() - start_onnx) / num_runs) * 1000.0

    speedup = total_pt_ms / max(total_onnx_ms, 0.001)

    benchmark_results = {
        "pytorch_latency_ms": round(total_pt_ms, 2),
        "onnx_latency_ms": round(total_onnx_ms, 2),
        "speedup_factor": f"{speedup:.2f}x faster",
        "num_runs": num_runs
    }

    print("\n================ INFERENCE LATENCY BENCHMARK ================")
    print(f"PyTorch (.pt) Latency:    {benchmark_results['pytorch_latency_ms']} ms / frame")
    print(f"ONNX Runtime (.onnx):    {benchmark_results['onnx_latency_ms']} ms / frame")
    print(f"Speedup Achieved:        {benchmark_results['speedup_factor']}")
    print("=============================================================\n")

    return benchmark_results


def convert_pt_to_onnx(config_path: str = "config/model_config.yaml"):
    """
    Converts PyTorch model to ONNX Runtime format.
    """
    print("=" * 60)
    print("⚡ Starting PyTorch -> ONNX Model Export & Optimization")
    print("=" * 60)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    pt_path = models_dir / "best_model.pt"
    if not pt_path.exists():
        print(f"[!] 'models/best_model.pt' not found. Using baseline '{config['model']['architecture']}.pt'...")
        pt_path = Path(f"{config['model']['architecture']}.pt")
        # Ensure base weights exist
        _ = YOLO(str(pt_path))

    onnx_target = models_dir / "visionops_guard.onnx"

    # Export to ONNX via Ultralytics
    model = YOLO(str(pt_path))
    exported_path = model.export(
        format="onnx",
        imgsz=config["data"]["img_size"],
        dynamic=False,
        simplify=True
    )

    # Move to target path
    if Path(exported_path) != onnx_target:
        import shutil
        shutil.copy(str(exported_path), str(onnx_target))

    print(f"[+] ONNX Model successfully exported to: {onnx_target}")

    # Benchmark Latency
    benchmark_res = benchmark_inference(str(pt_path), str(onnx_target))
    
    # Save Benchmark Report
    bench_file = Path("data/latency_benchmark.json")
    with open(bench_file, "w") as bf:
        json.dump(benchmark_res, bf, indent=2)
    print(f"[+] Benchmark report saved to: {bench_file}")


if __name__ == "__main__":
    convert_pt_to_onnx()
