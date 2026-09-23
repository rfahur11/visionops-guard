"""
VisionOps Guard - Model Training Pipeline
Trains Ultralytics YOLO Safety PPE Model with GPU Acceleration
and logs parameters, metrics & artifacts.
"""

import os
from pathlib import Path
import yaml
import torch
from ultralytics import YOLO

# Optional MLflow tracking
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except Exception:
    MLFLOW_AVAILABLE = False


# Enable headless Agg backend for matplotlib before importing font_manager
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.font_manager
    MATPLOTLIB_OK = True
except Exception:
    MATPLOTLIB_OK = False
    try:
        import ultralytics.utils.checks
        import ultralytics.data.utils
        ultralytics.utils.checks.check_font = lambda *args, **kwargs: None
        ultralytics.data.utils.check_font = lambda *args, **kwargs: None
    except Exception:
        pass




def train_model(config_path: str = "config/model_config.yaml"):
    """
    Executes training loop with GPU acceleration and optional MLflow tracking.
    """
    print("=" * 60)
    print("🚀 Starting VisionOps Guard Model Training Pipeline")
    print("=" * 60)

    # 1. Resolve Root Working Directory
    cfg_file = Path(config_path)
    if not cfg_file.exists():
        # Fallback if executed from inside src/models
        root_dir = Path(__file__).resolve().parents[2]
        os.chdir(root_dir)
        cfg_file = Path(config_path).resolve()

    with open(cfg_file, "r") as f:
        config = yaml.safe_load(f)

    # 2. Check GPU Acceleration
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[*] Hardware Acceleration Device: {device}")
    if torch.cuda.is_available():
        print(f"[*] Active GPU: {torch.cuda.get_device_name(0)}")

    epochs = config["model"]["epochs"]
    batch_size = config["model"]["batch_size"]
    img_size = config["data"]["img_size"]
    arch = config["model"]["architecture"]
    lr = config["model"].get("learning_rate", 0.005)
    optimizer_name = config["model"].get("optimizer", "AdamW")
    weight_decay = config["model"].get("weight_decay", 0.0005)

    # 3. Setup MLflow Tracking (Docker server or local fallback)
    if MLFLOW_AVAILABLE:
        try:
            import urllib.request
            try:
                urllib.request.urlopen("http://localhost:5000", timeout=2)
                mlflow.set_tracking_uri("http://localhost:5000")
                print("[*] Connected to MLflow Tracking Server at http://localhost:5000")
            except Exception:
                mlflow.set_tracking_uri("file:./mlruns")
                print("[*] Using local file tracking URI: ./mlruns")

            mlflow.set_experiment(config["project"]["name"])
            mlflow.start_run(run_name=f"YOLO26_PPE_Training_{arch}")
            mlflow.log_params({
                "architecture": arch,
                "epochs": epochs,
                "batch_size": batch_size,
                "img_size": img_size,
                "learning_rate": lr,
                "optimizer": optimizer_name,
                "weight_decay": weight_decay,
                "device": device
            })
            print("[+] MLflow tracking session initialized.")
        except Exception as e:
            print(f"[!] Warning initializing MLflow: {e}")

    # 4. Load Pre-trained Backbone
    print(f"[*] Loading pre-trained backbone: '{arch}.pt'...")
    model = YOLO(f"{arch}.pt")

    # 5. Launch Training Loop
    dataset_yaml = Path(config["data"]["dataset_yaml"]).resolve()
    project_dir = Path("runs/train").resolve()
    
    results = model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        lr0=lr,
        optimizer=optimizer_name,
        weight_decay=weight_decay,
        device=0 if torch.cuda.is_available() else "cpu",
        project=str(project_dir),
        name="visionops_ppe",
        exist_ok=True,
        plots=MATPLOTLIB_OK,
        verbose=True
    )

    # 6. Extract Best Weights & Save Artifacts
    best_pt = project_dir / "visionops_ppe" / "weights" / "best.pt"
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    target_pt = models_dir / "best_model.pt"
    
    if best_pt.exists():
        import shutil
        shutil.copy(str(best_pt), str(target_pt))
        print(f"[+] Best PyTorch weights saved to: {target_pt}")
        if MLFLOW_AVAILABLE:
            try:
                mlflow.log_artifact(str(target_pt), artifact_path="models")
            except Exception:
                pass

        # 7. Auto-Export to ONNX FP16
        print("[*] Exporting champion model to ONNX FP16...")
        try:
            onnx_path = models_dir / "visionops_guard.onnx"
            best_model = YOLO(str(target_pt))
            exported_file = best_model.export(
                format="onnx",
                imgsz=img_size,
                half=True,
                dynamic=False,
                simplify=True
            )
            if Path(exported_file).exists() and Path(exported_file) != onnx_path:
                shutil.copy(str(exported_file), str(onnx_path))
            print(f"[+] Exported ONNX FP16 saved to: {onnx_path}")
            if MLFLOW_AVAILABLE:
                try:
                    mlflow.log_artifact(str(onnx_path), artifact_path="models")
                except Exception:
                    pass
        except Exception as e:
            print(f"[!] Warning exporting ONNX: {e}")

    # 8. Log Final Evaluation Metrics to MLflow
    if results and hasattr(results, "results_dict"):
        rd = results.results_dict
        print("\n" + "=" * 60)
        print("📊 Training Results Summary:")
        print(f"   mAP50(B)    : {rd.get('metrics/mAP50(B)', 0):.4f}")
        print(f"   mAP50-95(B) : {rd.get('metrics/mAP50-95(B)', 0):.4f}")
        print(f"   Precision(B): {rd.get('metrics/precision(B)', 0):.4f}")
        print(f"   Recall(B)   : {rd.get('metrics/recall(B)', 0):.4f}")
        print("=" * 60)
        if MLFLOW_AVAILABLE:
            try:
                import re
                for k, v in rd.items():
                    if isinstance(v, (int, float)):
                        clean_key = re.sub(r"[^a-zA-Z0-9_\-\. :]", "_", k.replace("/", "_"))
                        mlflow.log_metric(clean_key, float(v))
            except Exception as e:
                print(f"[!] Error logging metrics to MLflow: {e}")

    if MLFLOW_AVAILABLE:
        try:
            mlflow.end_run()
        except Exception:
            pass

    print("[+] Training Pipeline completed successfully!")


if __name__ == "__main__":
    train_model()

