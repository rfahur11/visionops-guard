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

    # 3. Setup MLflow Tracking if available
    if MLFLOW_AVAILABLE:
        try:
            mlflow.set_experiment(config["project"]["name"])
            mlflow.start_run(run_name=f"YOLO_PPE_Training_{arch}")
            mlflow.log_params({
                "architecture": arch,
                "epochs": epochs,
                "batch_size": batch_size,
                "img_size": img_size,
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

    if MLFLOW_AVAILABLE:
        try:
            mlflow.end_run()
        except Exception:
            pass

    print("[+] Training Pipeline completed successfully!")


if __name__ == "__main__":
    train_model()

