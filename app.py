"""
VisionOps Guard - Hugging Face Spaces Entry Point
Launches Gradio UI for Safety PPE Inspection.
Compatible with CPU and Hugging Face ZeroGPU.
"""

import base64
import json
from pathlib import Path
import cv2
import numpy as np
import gradio as gr

# Monkeypatch upstream Gradio / gradio_client bug with boolean JSON schemas
try:
    import gradio_client.utils as client_utils

    _orig_get_type = getattr(client_utils, "get_type", None)
    if _orig_get_type:
        def _safe_get_type(schema):
            if not isinstance(schema, (dict, list)):
                return "Any"
            return _orig_get_type(schema)
        client_utils.get_type = _safe_get_type

    _orig_json_schema = getattr(client_utils, "_json_schema_to_python_type", None)
    if _orig_json_schema:
        def _safe_json_schema(schema, defs=None):
            if not isinstance(schema, dict):
                return "Any"
            if isinstance(schema.get("additionalProperties"), bool):
                schema = dict(schema)
                schema["additionalProperties"] = {}
            return _orig_json_schema(schema, defs)
        client_utils._json_schema_to_python_type = _safe_json_schema
except Exception as e:
    print(f"Warning: could not patch gradio_client: {e}")

from src.serving.predictor import SafetyPPEPredictor

# Optional ZeroGPU support if Space has GPU assigned
try:
    import spaces
    gpu_decorator = spaces.GPU
except ImportError:
    def gpu_decorator(fn):
        return fn

# Initialize Predictor Singleton
predictor = SafetyPPEPredictor()


@gpu_decorator
def inspect_safety_ppe(input_image: np.ndarray, conf_threshold: float = 0.45):
    """
    Gradio prediction function.
    """
    if input_image is None:
        return None, "Please upload an image or capture from webcam."

    # Convert RGB to BGR for OpenCV encoding
    img_bgr = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".jpg", img_bgr)
    img_bytes = buffer.tobytes()

    # Update confidence threshold
    predictor.conf_threshold = conf_threshold

    # Run Prediction
    result = predictor.predict(img_bytes)

    # Decode base64 preview image back to RGB
    img_data = base64.b64decode(result["annotated_image_base64"])
    nparr = np.frombuffer(img_data, np.uint8)
    ann_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    ann_rgb = cv2.cvtColor(ann_bgr, cv2.COLOR_BGR2RGB)

    # Format summary dictionary
    summary = {
        "status": "COMPLIANT ✅" if result["is_compliant"] else "VIOLATION DETECTED ⚠️",
        "violations_count": result["violations_count"],
        "inference_latency_ms": round(result["latency_ms"], 2),
        "detections_count": len(result["detections"]),
        "detected_objects": [
            {
                "class": d["class_name"].upper(),
                "confidence": f"{d['confidence']*100:.1f}%",
                "box_xyxy": d["box_xyxy"]
            }
            for d in result["detections"]
        ]
    }

    formatted_json_str = json.dumps(summary, indent=2)
    return ann_rgb, formatted_json_str


# Create Gradio Blocks Interface
with gr.Blocks(title="VisionOps Guard - Safety PPE AI", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🛡️ VisionOps Guard — Real-Time Safety PPE Inspection")
    gr.Markdown(
        "Upload a construction or industrial factory workplace image to inspect hardhat and safety vest compliance in real-time using ONNX Runtime."
    )

    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="numpy", label="Upload Image / Webcam Capture")
            conf_slider = gr.Slider(minimum=0.1, maximum=1.0, value=0.45, step=0.05, label="Confidence Threshold")
            submit_btn = gr.Button("Inspect Safety Compliance 🚀", variant="primary")
        
        with gr.Column():
            output_img = gr.Image(type="numpy", label="ONNX Detection Result")
            output_details = gr.Textbox(label="Compliance Analytics & Bounding Boxes", lines=12)

    submit_btn.click(
        fn=inspect_safety_ppe,
        inputs=[input_img, conf_slider],
        outputs=[output_img, output_details],
        api_name=False
    )


if __name__ == "__main__":
    demo.launch()
