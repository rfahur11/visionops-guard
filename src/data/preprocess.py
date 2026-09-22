"""
VisionOps Guard - Preprocessing & Augmentation Pipeline
Standardizes visual inputs to 640x640 using OpenCV, applies geometric & photometric augmentations,
and generates statistical reference features for MLOps Data Drift Monitoring.
"""

import random
import shutil
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import yaml


def apply_opencv_augmentations(img_rgb: np.ndarray, bboxes: list, img_size: int = 640):
    """
    Applies pure OpenCV data augmentations (flip, brightness, contrast, blur)
    and updates YOLO bounding box coordinates accordingly.
    """
    h, w, _ = img_rgb.shape
    # 1. Resize to target img_size
    img_resized = cv2.resize(img_rgb, (img_size, img_size))

    aug_img = img_resized.copy()
    aug_bboxes = []

    # 2. Random Horizontal Flip (p=0.5)
    do_flip = random.random() < 0.5
    if do_flip:
        aug_img = cv2.flip(aug_img, 1)

    for bbox in bboxes:
        cls_id, xc, yc, bw, bh = bbox
        if do_flip:
            xc = 1.0 - xc
        aug_bboxes.append((cls_id, xc, yc, bw, bh))

    # 3. Random Brightness & Contrast (p=0.4)
    if random.random() < 0.4:
        alpha = random.uniform(0.8, 1.2)  # Contrast control
        beta = random.randint(-25, 25)    # Brightness control
        aug_img = np.clip(alpha * aug_img + beta, 0, 255).astype(np.uint8)

    # 4. Random Gaussian Blur / Motion Blur (p=0.2)
    if random.random() < 0.2:
        kernel_size = random.choice([3, 5])
        aug_img = cv2.GaussianBlur(aug_img, (kernel_size, kernel_size), 0)

    return aug_img, aug_bboxes


def compute_image_metrics(img_rgb: np.ndarray) -> dict:
    """
    Computes statistical visual features used for Data Drift monitoring.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    mean_brightness = float(np.mean(gray))
    std_brightness = float(np.std(gray))
    contrast = float(gray.max() - gray.min())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    return {
        "mean_brightness": mean_brightness,
        "std_brightness": std_brightness,
        "contrast": contrast,
        "sharpness": sharpness
    }


def preprocess_dataset(
    raw_dir: str = "data/raw", 
    processed_dir: str = "data/processed",
    config_path: str = "config/model_config.yaml"
):
    """
    Executes full preprocessing and feature extraction pipeline.
    """
    print(f"[*] Starting Data Preprocessing from '{raw_dir}' -> '{processed_dir}'")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    img_size = config["data"]["img_size"]
    raw_images_path = Path(raw_dir) / "images"
    raw_labels_path = Path(raw_dir) / "labels"
    proc_images_path = Path(processed_dir) / "images"
    proc_labels_path = Path(processed_dir) / "labels"

    # Clean existing processed directory
    if Path(processed_dir).exists():
        shutil.rmtree(processed_dir, ignore_errors=True)

    reference_features = []
    splits = ["train", "val", "test"]

    for split in splits:
        raw_img_split = raw_images_path / split
        raw_lbl_split = raw_labels_path / split
        out_img_split = proc_images_path / split
        out_lbl_split = proc_labels_path / split

        out_img_split.mkdir(parents=True, exist_ok=True)
        out_lbl_split.mkdir(parents=True, exist_ok=True)

        if not raw_img_split.exists():
            continue

        for img_file in raw_img_split.glob("*.jpg"):
            lbl_file = raw_lbl_split / f"{img_file.stem}.txt"
            img = cv2.imread(str(img_file))
            if img is None:
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            bboxes = []
            if lbl_file.exists():
                with open(lbl_file, "r") as lf:
                    for line in lf:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            cls_id = int(parts[0])
                            xc, yc, w, h = map(float, parts[1:])
                            bboxes.append((cls_id, xc, yc, w, h))

            if split == "train":
                proc_img, proc_bboxes = apply_opencv_augmentations(img_rgb, bboxes, img_size)
            else:
                proc_img = cv2.resize(img_rgb, (img_size, img_size))
                proc_bboxes = bboxes

            # Save Processed Image (convert RGB back to BGR for OpenCV write)
            out_img_file = out_img_split / img_file.name
            cv2.imwrite(str(out_img_file), cv2.cvtColor(proc_img, cv2.COLOR_RGB2BGR))

            # Save Processed YOLO Labels
            out_lbl_file = out_lbl_split / f"{img_file.stem}.txt"
            with open(out_lbl_file, "w") as of:
                for cls_id, xc, yc, bw, bh in proc_bboxes:
                    of.write(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

            # Extract statistical features for Training baseline (Reference set)
            if split == "train":
                metrics = compute_image_metrics(proc_img)
                metrics["filename"] = img_file.name
                reference_features.append(metrics)

    # Save reference statistical features for Evidently AI Drift Detection
    if reference_features:
        ref_df = pd.DataFrame(reference_features)
        ref_path = Path(processed_dir) / "reference_features.csv"
        ref_df.to_csv(ref_path, index=False)
        print(f"[+] Saved baseline reference features ({len(ref_df)} rows) to: {ref_path}")

    print(f"[+] Preprocessing completed successfully for all splits!")


if __name__ == "__main__":
    preprocess_dataset()
