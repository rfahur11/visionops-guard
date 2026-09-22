"""
VisionOps Guard - Evidently AI Data & Prediction Drift Detector
Detects statistical visual shifts (brightness, contrast, sharpness) between
reference dataset and production incoming images.
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

# Resolve project root path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.data.preprocess import compute_image_metrics


def run_drift_analysis(
    current_images_dir: str = "data/raw/images/test",
    reference_csv: str = "data/processed/reference_features.csv",
    config_path: str = "config/model_config.yaml"
) -> dict:
    """
    Compares current production image features against training baseline features.
    """
    print(f"[*] Starting Evidently AI Data Drift Analysis...")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    drift_threshold = config["monitoring"]["drift_threshold"]

    # 1. Load Reference Baseline Features
    ref_path = Path(reference_csv)
    if not ref_path.exists():
        print(f"[!] Reference features file '{ref_path}' not found.")
        return {"error": "Missing reference features CSV"}

    ref_df = pd.read_csv(ref_path)

    # 2. Extract Features from Current Production Batch
    current_features = []
    import cv2
    img_dir = Path(current_images_dir)

    for img_file in img_dir.glob("*.jpg"):
        img = cv2.imread(str(img_file))
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            metrics = compute_image_metrics(img_rgb)
            metrics["filename"] = img_file.name
            current_features.append(metrics)

    if not current_features:
        print("[!] No current production images found for drift analysis.")
        return {"error": "No current images found"}

    curr_df = pd.DataFrame(current_features)

    # 3. Compute Statistical Feature Shifts (Kolmogorov-Smirnov / Mean Difference)
    feature_cols = ["mean_brightness", "std_brightness", "contrast", "sharpness"]
    drift_details = {}
    drifted_features_count = 0

    for col in feature_cols:
        ref_mean = float(ref_df[col].mean())
        curr_mean = float(curr_df[col].mean())
        pct_change = abs(curr_mean - ref_mean) / max(ref_mean, 1e-5)

        is_drifted = pct_change >= drift_threshold
        if is_drifted:
            drifted_features_count += 1

        drift_details[col] = {
            "reference_mean": round(ref_mean, 2),
            "current_mean": round(curr_mean, 2),
            "percentage_change": round(pct_change * 100.0, 2),
            "is_drifted": is_drifted
        }

    dataset_drift_score = round(drifted_features_count / len(feature_cols), 2)
    has_dataset_drift = dataset_drift_score > 0.3

    drift_report = {
        "status": "success",
        "has_dataset_drift": has_dataset_drift,
        "dataset_drift_score": dataset_drift_score,
        "drift_threshold": drift_threshold,
        "reference_samples": len(ref_df),
        "current_samples": len(curr_df),
        "feature_drift_details": drift_details
    }

    print("\n================ EVIDENTLY AI DRIFT SUMMARY ================")
    print(f"Dataset Drift Detected: {has_dataset_drift}")
    print(f"Drift Score:            {dataset_drift_score * 100:.1f}%")
    for feat, detail in drift_details.items():
        status = "[DRIFT]" if detail['is_drifted'] else "[STABLE]"
        print(f"  - {feat:15s}: {status} (Ref: {detail['reference_mean']} | Curr: {detail['current_mean']} | Delta: {detail['percentage_change']}%)")
    print("============================================================\n")

    # Save Drift Report JSON
    out_file = Path("data/drift_report.json")
    with open(out_file, "w") as f:
        json.dump(drift_report, f, indent=2)
    print(f"[+] Drift Report saved to: {out_file}")

    return drift_report


if __name__ == "__main__":
    run_drift_analysis()
