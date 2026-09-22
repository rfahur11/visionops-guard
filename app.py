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
        "status": "COMPLIANT" if result["is_compliant"] else "VIOLATION DETECTED",
        "violations_count": result["violations_count"],
        "inference_latency_ms": result["latency_ms"],
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
with gr.Blocks(title="VisionOps Guard - Safety PPE AI") as demo:
    gr.Markdown("# 🛡️ VisionOps Guard — Real-Time Safety PPE Inspection")
    gr.Markdown("Upload a construction/factory image or use your webcam to inspect hardhat and safety vest compliance.")

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
        outputs=[output_img, output_details]
    )


if __name__ == "__main__":
    demo.launch()
