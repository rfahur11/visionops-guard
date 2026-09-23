---
name: mlflow-model-management
description: Manages MLflow model training experiments, metrics comparison (mAP50, Precision, Recall), run analysis, and model registry promotions for VisionOps Guard.
---

# MLflow Model Management Skill

This skill provides protocols and automated workflows for tracking experiments, comparing model architectures, selecting Champion models, and promoting models in the MLflow Model Registry for VisionOps Guard.

## Tracking Server Access
- **Local Tracking UI:** `http://localhost:5000`
- **Docker Service:** `visionops-mlflow`
- **Artifacts Store:** Local directory `./mlruns` mounted into container.

---

## 1. Quick Experiment Inspection via Python SDK

To quickly inspect or query recent training runs without opening a browser:

```python
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()

# Get experiment by name
experiment = client.get_experiment_by_name("visionops_ppe")
if experiment:
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.metrics/mAP50(B) DESC"],
        max_results=5
    )
    for r in runs:
        print(f"Run ID: {r.info.run_id}")
        print(f"mAP50: {r.data.metrics.get('metrics/mAP50(B)', 'N/A')}")
        print(f"Precision: {r.data.metrics.get('metrics/precision(B)', 'N/A')}")
```

---

## 2. Champion Model Promotion Protocol

When a new training run achieves higher accuracy metrics:
1. Verify `mAP50` >= 0.75 and `Precision` >= 0.85.
2. Register the model artifact:
   ```python
   model_uri = f"runs:/{run_id}/weights/best.pt"
   mlflow.register_model(model_uri, "visionops_guard_model")
   ```
3. Transition stage in Model Registry:
   ```python
   client.transition_model_version_stage(
       name="visionops_guard_model",
       version=1,
       stage="Production"
   )
   ```
4. Export the champion model to ONNX FP16 (`models/visionops_guard.onnx`) for production serving.

---

## 3. Best Practices
- Keep training on the native host (`.venv`) for direct GPU hardware acceleration.
- Let MLflow Docker service read the `./mlruns` bind mount for persistent, zero-duplicate storage.
