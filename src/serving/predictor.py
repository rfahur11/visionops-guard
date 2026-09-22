"""
VisionOps Guard - ONNX Runtime Inference Engine
High-performance CVOps predictor for real-time safety PPE inspection.
"""

import base64
import time
from pathlib import Path
import cv2
import numpy as np
import yaml
import onnxruntime as ort


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
            self.bgr_colors[name] = (rgb[2], rgb[1], rgb[0]) # BGR

        # Model Loading
        model_path = Path(self.config["model"]["onnx_export_path"])
        if not model_path.exists():
            model_path = Path("models/visionops_guard.onnx")

        if not model_path.exists():
            print(f"[!] '{model_path}' not found on disk. Auto-generating ONNX model...")
            model_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                from ultralytics import YOLO
                base_yolo = YOLO("yolov8n.pt")
                exported = base_yolo.export(format="onnx", imgsz=self.img_size, simplify=True)
                import shutil
                shutil.copy(exported, str(model_path))
                print(f"[+] Fallback ONNX model created at: {model_path}")
            except Exception as e:
                print(f"[!] Fallback creation error: {e}")

        print(f"[*] Loading ONNX Runtime Session: '{model_path}'")
        try:
            self.session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
        except Exception as e:
            print(f"[!] Error loading ONNX session: {e}")
            self.session = ort.InferenceSession(str(model_path))

        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    def preprocess(self, img_bytes: bytes):
        """
        Decodes raw bytes and resizes to target input shape (1, 3, 640, 640).
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

    def postprocess(self, outputs, orig_w: int, orig_h: int):
        """
        Parses YOLO ONNX output tensor shape [1, 9, 8400] and extracts detections.
        """
        preds = outputs[0]
        if len(preds.shape) == 3:
            preds = preds[0]  # (9, 8400) or (8400, 9)

        if preds.shape[0] < preds.shape[1]:
            preds = preds.T  # Transpose to (8400, 9) -> [xc, yc, w, h, cls0, cls1, cls2, cls3, cls4]

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
                # Convert center xywh to pixel corner box xywh
                x1 = int((xc - w / 2) * scale_x)
                y1 = int((yc - h / 2) * scale_y)
                box_w = int(w * scale_x)
                box_h = int(h * scale_y)

                boxes.append([x1, y1, box_w, box_h])
                confidences.append(confidence)
                class_ids.append(class_id)

        # Apply Non-Maximum Suppression (NMS)
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

    def predict(self, img_bytes: bytes) -> dict:
        """
        Main inference entry point.
        """
        start_t = time.time()
        orig_img, input_tensor, orig_w, orig_h = self.preprocess(img_bytes)
        
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        detections = self.postprocess(outputs, orig_w, orig_h)
        
        latency_ms = round((time.time() - start_t) * 1000.0, 2)

        # Generate base64 annotated preview
        annotated_img = self.draw_detections(orig_img, detections)
        _, buffer = cv2.imencode(".jpg", annotated_img)
        base64_preview = base64.b64encode(buffer).decode("utf-8")

        # Compliance Check Summary
        violations = [d for d in detections if d["class_name"] in ["no_hardhat", "no_vest"]]
        is_compliant = len(violations) == 0

        return {
            "status": "success",
            "is_compliant": is_compliant,
            "violations_count": len(violations),
            "detections": detections,
            "latency_ms": latency_ms,
            "annotated_image_base64": base64_preview
        }
