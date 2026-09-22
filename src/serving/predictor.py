"""
VisionOps Guard - Hybrid ONNX Runtime & PyTorch Inference Engine
High-performance CVOps predictor with dual-mode fallback (ONNX Runtime -> PyTorch YOLO).
"""

import base64
import time
from pathlib import Path
import cv2
import numpy as np
import yaml
from ultralytics import YOLO

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False


class SafetyPPEPredictor:
    def __init__(self, config_path: str = "config/model_config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.img_size = self.config["data"]["img_size"]
        self.conf_threshold = self.config["inference"]["conf_threshold"]
        self.iou_threshold = self.config["inference"]["iou_threshold"]
        self.classes = self.config["data"]["classes"]
        self.class_colors = self.config["data"]["class_colors"]

        # Color mapping (HEX to BGR)
        self.bgr_colors = {}
        for name, hex_code in self.class_colors.items():
            hex_str = hex_code.lstrip("#")
            rgb = tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
            self.bgr_colors[name] = (rgb[2], rgb[1], rgb[0])  # BGR

        # Model Loading Logic
        self.session = None
        self.pt_model = None

        model_path = Path(self.config["model"]["onnx_export_path"])
        if not model_path.exists():
            model_path = Path("models/visionops_guard.onnx")

        # Try loading ONNX model first
        if ORT_AVAILABLE and model_path.exists():
            try:
                print(f"[*] Loading ONNX Runtime Session: '{model_path}'")
                self.session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
                self.input_name = self.session.get_inputs()[0].name
                self.output_names = [o.name for o in self.session.get_outputs()]
                print("[+] ONNX Runtime Session initialized successfully.")
            except Exception as e:
                print(f"[!] ONNX Runtime initialization failed: {e}")
                self.session = None

        # Fallback to PyTorch YOLO model if ONNX is not available
        if self.session is None:
            print("[*] Falling back to native PyTorch YOLO Engine...")
            pt_path = Path("models/best_model.pt")
            if not pt_path.exists():
                pt_path = Path("yolov8n.pt")
            self.pt_model = YOLO(str(pt_path))
            print(f"[+] PyTorch YOLO model loaded from: {pt_path}")

    def preprocess(self, img_bytes: bytes):
        """
        Decodes raw bytes and resizes to target input shape.
        """
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image bytes.")

        orig_h, orig_w = img.shape[:2]
        img_resized = cv2.resize(img, (self.img_size, self.img_size))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_float = img_rgb.astype(np.float32) / 255.0
        img_chw = np.transpose(img_float, (2, 0, 1))
        input_tensor = np.expand_dims(img_chw, axis=0)

        return img, input_tensor, orig_w, orig_h

    def postprocess_onnx(self, outputs, orig_w: int, orig_h: int):
        """
        Parses YOLO ONNX output tensor and extracts detections.
        """
        preds = outputs[0]
        if len(preds.shape) == 3:
            preds = preds[0]

        if preds.shape[0] < preds.shape[1]:
            preds = preds.T

        boxes = []
        confidences = []
        class_ids = []

        scale_x = orig_w / float(self.img_size)
        scale_y = orig_h / float(self.img_size)

        for row in preds:
            scores = row[4:]
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])

            if confidence >= self.conf_threshold:
                xc, yc, w, h = row[0:4]
                x1 = int((xc - w / 2) * scale_x)
                y1 = int((yc - h / 2) * scale_y)
                box_w = int(w * scale_x)
                box_h = int(h * scale_y)

                boxes.append([x1, y1, box_w, box_h])
                confidences.append(confidence)
                class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.iou_threshold)
        detections = []
        if len(indices) > 0:
            flat_indices = indices.flatten() if isinstance(indices, np.ndarray) else indices
            for i in flat_indices:
                box = boxes[i]
                cls_id = class_ids[i]
                cls_name = self.classes.get(cls_id, f"class_{cls_id}")
                detections.append({
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": round(float(confidences[i]), 4),
                    "box_xywh": box,
                    "box_xyxy": [box[0], box[1], box[0] + box[2], box[1] + box[3]]
                })
        return detections

    def predict_pytorch(self, orig_img: np.ndarray):
        """
        Inference using native PyTorch YOLO model fallback.
        """
        results = self.pt_model(orig_img, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)[0]
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            xyxy = [int(x) for x in box.xyxy[0].tolist()]
            xywh = [xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]]
            cls_name = self.classes.get(cls_id, f"class_{cls_id}")
            detections.append({
                "class_id": cls_id,
                "class_name": cls_name,
                "confidence": round(conf, 4),
                "box_xywh": xywh,
                "box_xyxy": xyxy
            })
        return detections

    def draw_detections(self, orig_img: np.ndarray, detections: list) -> np.ndarray:
        """
        Annotates image with colored bounding boxes and labels.
        """
        annotated = orig_img.copy()
        for det in detections:
            cls_name = det["class_name"]
            conf = det["confidence"]
            x1, y1, x2, y2 = det["box_xyxy"]

            color = self.bgr_colors.get(cls_name, (0, 255, 0))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label_text = f"{cls_name} {conf:.2f}"
            (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(annotated, (x1, y1 - text_h - 6), (x1 + text_w + 4, y1), color, -1)
            cv2.putText(annotated, label_text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return annotated

    def predict(self, img_bytes: bytes, lang: str = "id") -> dict:
        """
        Main inference entry point. Supports ONNX Runtime with PyTorch YOLO fallback.
        Supports bilingual output ('id' for Indonesian, 'en' for English).
        """
        start_t = time.time()
        orig_img, input_tensor, orig_w, orig_h = self.preprocess(img_bytes)

        if self.session is not None:
            outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
            detections = self.postprocess_onnx(outputs, orig_w, orig_h)
        else:
            detections = self.predict_pytorch(orig_img)

        latency_ms = round((time.time() - start_t) * 1000.0, 2)

        # Generate base64 annotated preview
        annotated_img = self.draw_detections(orig_img, detections)
        _, buffer = cv2.imencode(".jpg", annotated_img)
        base64_preview = base64.b64encode(buffer).decode("utf-8")

        # Smart Industrial PPE Compliance Assessment Logic
        is_en = str(lang).lower().startswith("en")
        persons = [d for d in detections if d["class_name"] == "person"]
        hardhats = [d for d in detections if d["class_name"] == "hardhat"]
        vests = [d for d in detections if d["class_name"] == "vest"]
        no_hardhats = [d for d in detections if d["class_name"] == "no_hardhat"]
        no_vests = [d for d in detections if d["class_name"] == "no_vest"]

        # Bilingual display names for detected classes
        class_display_map_id = {
            "hardhat": "Helm Proyek (Hardhat)",
            "no_hardhat": "Tanpa Helm (No Hardhat)",
            "vest": "Rompi Safety (Vest)",
            "no_vest": "Tanpa Rompi (No Vest)",
            "person": "Pekerja (Person)"
        }
        class_display_map_en = {
            "hardhat": "Safety Hardhat",
            "no_hardhat": "No Hardhat (Violation)",
            "vest": "Safety Vest",
            "no_vest": "No Vest (Violation)",
            "person": "Worker / Person"
        }

        for d in detections:
            d["class_display"] = (
                class_display_map_en.get(d["class_name"], d["class_name"].upper())
                if is_en else
                class_display_map_id.get(d["class_name"], d["class_name"].upper())
            )

        violation_details = []
        if len(detections) == 0:
            compliance_status = "NO PERSON / PPE DETECTED" if is_en else "TIDAK ADA PEKERJA / APD TERDETEKSI"
            is_compliant = False
            violations_count = 0
            assessment = (
                "No worker or safety PPE detected in frame. Please adjust camera or lower confidence threshold."
                if is_en else
                "Tidak ada pekerja atau atribut APD yang terdeteksi dalam frame. Silakan turunkan threshold atau arahkan kamera ke pekerja."
            )
        elif len(persons) == 0 and (len(hardhats) > 0 or len(vests) > 0):
            compliance_status = "PARTIAL PPE DETECTED" if is_en else "APD SEBAGIAN TERDETEKSI"
            is_compliant = True
            violations_count = 0
            assessment = (
                "Safety PPE items (hardhat / vest) detected in workspace."
                if is_en else
                "Atribut APD (helm / rompi) terdeteksi di area kerja."
            )
        else:
            # When person is detected:
            if len(no_hardhats) > 0 or len(hardhats) == 0:
                violation_details.append(
                    "Worker Missing Safety Hardhat" if is_en else "Pekerja Tidak Memakai Helm Proyek (Missing Hardhat)"
                )
            if len(no_vests) > 0 or len(vests) == 0:
                violation_details.append(
                    "Worker Missing High-Visibility Vest" if is_en else "Pekerja Tidak Memakai Rompi Safety (Missing Safety Vest)"
                )

            violations_count = len(violation_details)
            if violations_count > 0:
                compliance_status = "VIOLATION DETECTED" if is_en else "PELANGGARAN K3 TERDETEKSI"
                is_compliant = False
                assessment = (
                    f"Safety Non-Compliance: Detected {len(persons)} worker(s) missing required PPE: {'; '.join(violation_details)}."
                    if is_en else
                    f"Peringatan K3: Terdeteksi {len(persons)} pekerja tanpa APD lengkap: {'; '.join(violation_details)}."
                )
            else:
                compliance_status = "COMPLIANT" if is_en else "PATUH STANDAR K3"
                is_compliant = True
                assessment = (
                    f"Safety Standards Met: {len(persons)} worker(s) fully equipped with PPE (Hardhat & Vest)."
                    if is_en else
                    f"Standar K3 Terpenuhi: {len(persons)} pekerja terdeteksi mengenakan APD lengkap (Helm & Rompi)."
                )

        return {
            "status": "success",
            "compliance_status": compliance_status,
            "is_compliant": is_compliant,
            "violations_count": violations_count,
            "violation_details": violation_details,
            "assessment": assessment,
            "detections": detections,
            "latency_ms": latency_ms,
            "annotated_image_base64": base64_preview
        }
