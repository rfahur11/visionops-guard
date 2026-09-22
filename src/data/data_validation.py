"""
VisionOps Guard - Data Quality Assurance (QA) Script
Validates image decodability, bbox coordinate bounds, file integrity, and class balance.
"""

import json
from pathlib import Path
import cv2
import yaml


def validate_yolo_dataset(data_dir: str = "data/raw", config_path: str = "config/model_config.yaml") -> dict:
    """
    Performs comprehensive QA checks on visual datasets.
    """
    print(f"[*] Starting Data Quality Assurance (QA) on: {data_dir}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    valid_classes = set(config["data"]["classes"].keys())
    valid_exts = tuple(config["data"]["valid_extensions"])

    report = {
        "dataset_path": data_dir,
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "splits": {},
        "class_counts": {name: 0 for name in config["data"]["classes"].values()},
        "total_images": 0,
        "total_annotations": 0
    }

    base_path = Path(data_dir)
    images_base = base_path / "images"
    labels_base = base_path / "labels"

    if not images_base.exists() or not labels_base.exists():
        report["is_valid"] = False
        report["errors"].append("Missing 'images/' or 'labels/' directories.")
        return report

    splits = [d.name for d in images_base.iterdir() if d.is_dir()]
    if not splits:
        splits = ["train", "val", "test"]

    for split in splits:
        img_dir = images_base / split
        lbl_dir = labels_base / split

        split_stat = {
            "image_count": 0,
            "corrupt_images": 0,
            "missing_labels": 0,
            "invalid_bboxes": 0
        }

        if not img_dir.exists():
            report["warnings"].append(f"Split {split} not found in images.")
            continue

        for img_file in img_dir.iterdir():
            if img_file.suffix.lower() not in valid_exts:
                continue

            split_stat["image_count"] += 1
            report["total_images"] += 1

            # 1. Verify image can be decoded
            img = cv2.imread(str(img_file))
            if img is None or img.size == 0:
                split_stat["corrupt_images"] += 1
                report["errors"].append(f"Corrupt image file: {img_file}")
                report["is_valid"] = False
                continue

            # 2. Check corresponding label file
            lbl_file = lbl_dir / f"{img_file.stem}.txt"
            if not lbl_file.exists():
                split_stat["missing_labels"] += 1
                report["warnings"].append(f"Missing label file for: {img_file.name}")
                continue

            # 3. Check label contents & bbox bounds
            with open(lbl_file, "r") as lf:
                lines = [l.strip() for l in lf.readlines() if l.strip()]

            for line in lines:
                parts = line.split()
                if len(parts) != 5:
                    split_stat["invalid_bboxes"] += 1
                    report["errors"].append(f"Invalid YOLO bbox format in {lbl_file.name}: '{line}'")
                    report["is_valid"] = False
                    continue

                try:
                    cls_id = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:])
                except ValueError:
                    split_stat["invalid_bboxes"] += 1
                    report["errors"].append(f"Non-numeric bbox values in {lbl_file.name}")
                    report["is_valid"] = False
                    continue

                # Check class validity
                if cls_id not in valid_classes:
                    split_stat["invalid_bboxes"] += 1
                    report["errors"].append(f"Unknown class ID {cls_id} in {lbl_file.name}")
                    report["is_valid"] = False

                # Check coordinate range [0, 1]
                if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                    split_stat["invalid_bboxes"] += 1
                    report["errors"].append(f"Bbox out-of-bounds in {lbl_file.name}: xc={xc}, yc={yc}, w={w}, h={h}")
                    report["is_valid"] = False

                cls_name = config["data"]["classes"].get(cls_id, f"unknown_{cls_id}")
                report["class_counts"][cls_name] = report["class_counts"].get(cls_name, 0) + 1
                report["total_annotations"] += 1

        report["splits"][split] = split_stat

    # Output QA Summary
    print("\n================ DATA QA SUMMARY ================")
    print(f"Data Validity Passed: {report['is_valid']}")
    print(f"Total Images: {report['total_images']}")
    print(f"Total Annotations: {report['total_annotations']}")
    print(f"Class Breakdown: {report['class_counts']}")
    print(f"Errors Found: {len(report['errors'])}")
    print(f"Warnings Found: {len(report['warnings'])}")
    print("=================================================\n")

    return report


if __name__ == "__main__":
    report = validate_yolo_dataset()
    # Save QA Report
    out_path = Path("data/qa_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[+] QA Report saved to: {out_path}")
