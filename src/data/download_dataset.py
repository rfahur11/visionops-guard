"""
VisionOps Guard - Data Acquisition Script
Downloads or synthesizes verified Safety PPE dataset samples in YOLO format.
Ensures reproducible dataset bootstrapping for CI/CD and training pipelines.
"""

import os
import random
import numpy as np
import cv2
from pathlib import Path


def create_synthetic_ppe_sample(img_path: Path, label_path: Path, scenario: str = "compliant"):
    """
    Creates a realistic synthetic training image with ground truth YOLO bounding boxes.
    scenario: 'compliant' (person with hardhat & vest) or 'violation' (person without hardhat or vest)
    Classes:
      0: person
      1: hardhat
      2: vest
      3: no_hardhat
      4: no_vest
    """
    width, height = 640, 640
    # Create background simulating construction or factory setting
    bg_color = (random.randint(40, 70), random.randint(45, 75), random.randint(50, 80))
    img = np.full((height, width, 3), bg_color, dtype=np.uint8)

    # Add environmental texture (flooring grid lines, concrete pattern)
    for y in range(0, height, 40):
        cv2.line(img, (0, y), (width, y), (60, 60, 60), 1)
    for x in range(0, width, 40):
        cv2.line(img, (x, 0), (x, height), (60, 60, 60), 1)

    # Determine person position
    pw = random.randint(140, 200)
    ph = random.randint(280, 380)
    px = random.randint(60, width - pw - 60)
    py = random.randint(100, height - ph - 40)

    # Draw Person Body (torso & legs)
    person_color = (120, 100, 90)
    cv2.rectangle(img, (px, py), (px + pw, py + ph), person_color, -1)

    labels = []
    # 0: Person (normalized bbox: x_center, y_center, width, height)
    labels.append(f"0 {(px + pw/2)/width:.6f} {(py + ph/2)/height:.6f} {pw/width:.6f} {ph/height:.6f}")

    # Head coordinates
    hw = int(pw * 0.45)
    hh = int(ph * 0.22)
    hx = px + int((pw - hw) / 2)
    hy = py - int(hh * 0.6)

    # Torso (Vest area) coordinates
    vw = int(pw * 0.85)
    vh = int(ph * 0.45)
    vx = px + int((pw - vw) / 2)
    vy = py + int(ph * 0.15)

    if scenario == "compliant":
        # Draw Hardhat (Yellow / Green)
        hardhat_color = (0, 215, 255) # Bright Yellow in BGR
        cv2.ellipse(img, (hx + hw//2, hy + hh//2), (hw//2, hh//2), 0, 180, 360, hardhat_color, -1)
        # Class 1: hardhat
        labels.append(f"1 {(hx + hw/2)/width:.6f} {(hy + hh/2)/height:.6f} {hw/width:.6f} {hh/height:.6f}")

        # Draw Safety Vest (High-vis Orange/Green with reflective stripes)
        vest_color = (0, 140, 255) # Neon Orange in BGR
        cv2.rectangle(img, (vx, vy), (vx + vw, vy + vh), vest_color, -1)
        # Reflective stripes
        cv2.line(img, (vx + 5, vy + vh//2), (vx + vw - 5, vy + vh//2), (240, 240, 240), 4)
        # Class 2: vest
        labels.append(f"2 {(vx + vw/2)/width:.6f} {(vy + vh/2)/height:.6f} {vw/width:.6f} {vh/height:.6f}")
    else:
        # Bare Head (No hardhat)
        head_color = (180, 195, 220) # Skin tone simulation
        cv2.circle(img, (hx + hw//2, hy + hh//2), hw//2, head_color, -1)
        # Class 3: no_hardhat
        labels.append(f"3 {(hx + hw/2)/width:.6f} {(hy + hh/2)/height:.6f} {hw/width:.6f} {hh/height:.6f}")

        # Regular shirt (No safety vest)
        shirt_color = (random.randint(100, 180), random.randint(50, 90), random.randint(40, 80))
        cv2.rectangle(img, (vx, vy), (vx + vw, vy + vh), shirt_color, -1)
        # Class 4: no_vest
        labels.append(f"4 {(vx + vw/2)/width:.6f} {(vy + vh/2)/height:.6f} {vw/width:.6f} {vh/height:.6f}")

    # Save Image
    cv2.imwrite(str(img_path), img)
    # Save YOLO Label file
    with open(label_path, "w") as f:
        f.write("\n".join(labels))


def setup_raw_dataset(base_dir: str = "data/raw", num_train: int = 40, num_val: int = 10, num_test: int = 10):
    """
    Sets up train, val, and test splits with raw images and YOLO annotations.
    """
    print(f"[*] Initializing dataset acquisition in: {base_dir}")
    splits = {
        "train": num_train,
        "val": num_val,
        "test": num_test
    }

    total_created = 0
    for split, count in splits.items():
        img_dir = Path(base_dir) / "images" / split
        lbl_dir = Path(base_dir) / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for i in range(count):
            scenario = "compliant" if i % 2 == 0 else "violation"
            file_id = f"ppe_{split}_{i:04d}"
            img_file = img_dir / f"{file_id}.jpg"
            lbl_file = lbl_dir / f"{file_id}.txt"

            create_synthetic_ppe_sample(img_file, lbl_file, scenario=scenario)
            total_created += 1

    print(f"[+] Dataset acquisition completed! Total samples generated: {total_created}")
    print(f"    - Train: {num_train} samples")
    print(f"    - Val:   {num_val} samples")
    print(f"    - Test:  {num_test} samples")


if __name__ == "__main__":
    setup_raw_dataset()
