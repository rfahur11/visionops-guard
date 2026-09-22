"""
VisionOps Guard - Hugging Face Spaces Entry Point
Launches Gradio UI for Safety PPE Inspection.
"""

import base64
import io
import cv2
import numpy as np
import gradio as gr
from src.serving.predictor import SafetyPPEPredictor

# Initialize Predictor Singleton
predictor = SafetyPPEPredictor()


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

    # Format JSON summary
    summary = {
        "is_compliant": result["is_compliant"],
        "violations_count": result["violations_count"],
        "inference_latency_ms": result["latency_ms"],
        "detections_count": len(result["detections"]),
        "detections": result["detections"]
    }

    return ann_rgb, summary


# Create Gradio Blocks Interface
with gr.Blocks(title="VisionOps Guard - Safety PPE AI", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🛡️ VisionOps Guard — Real-Time Safety PPE Inspection")
    gr.Markdown("Upload a construction/factory image or use your webcam to inspect hardhat and safety vest compliance.")

    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="numpy", label="Upload Image / Webcam Capture")
            conf_slider = gr.Slider(minimum=0.1, maximum=1.0, value=0.45, step=0.05, label="Confidence Threshold")
            submit_btn = gr.Button("Inspect Safety Compliance 🚀", variant="primary")
        
        with gr.Column():
            output_img = gr.Image(type="numpy", label="ONNX Detection Result")
            output_json = gr.JSON(label="Compliance Analytics & Bounding Boxes")

    submit_btn.click(
        fn=inspect_safety_ppe,
        inputs=[input_img, conf_slider],
        outputs=[output_img, output_json]
    )

if __name__ == "__main__":
    demo.launch()
