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
def inspect_safety_ppe(input_image: np.ndarray, conf_threshold: float = 0.20, progress=gr.Progress(track_tqdm=True)):
    """
    Gradio prediction function with real-time animated progress bar.
    """
    if input_image is None:
        return None, "⚠️ Belum ada foto yang dipilih. Silakan upload gambar atau klik tombol kamera untuk menjepret foto."

    progress(0.15, desc="📸 Mempersiapkan frame gambar...")

    # Convert RGB to BGR for OpenCV encoding
    img_bgr = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".jpg", img_bgr)
    img_bytes = buffer.tobytes()

    # Update confidence threshold
    predictor.conf_threshold = conf_threshold

    progress(0.50, desc="⚡ Menjalankan inferensi ONNX Runtime...")
    # Run Prediction
    result = predictor.predict(img_bytes)

    progress(0.85, desc="🛡️ Menganalisis kepatuhan standar K3 APD...")
    # Decode base64 preview image back to RGB
    img_data = base64.b64decode(result["annotated_image_base64"])
    nparr = np.frombuffer(img_data, np.uint8)
    ann_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    ann_rgb = cv2.cvtColor(ann_bgr, cv2.COLOR_BGR2RGB)

    # Format summary dictionary with clean Unicode & K3 details
    status_text = result.get("compliance_status", "COMPLIANT" if result["is_compliant"] else "VIOLATION DETECTED")
    if result["is_compliant"]:
        status_display = f"{status_text} ✅"
    elif status_text.startswith("NO"):
        status_display = f"{status_text} ℹ️"
    else:
        status_display = f"{status_text} ⚠️"

    summary = {
        "status": status_display,
        "k3_assessment": result.get("assessment", ""),
        "violations_count": result["violations_count"],
        "violation_details": result.get("violation_details", []),
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

    progress(1.0, desc="✅ Analisis Selesai!")
    formatted_json_str = json.dumps(summary, indent=2, ensure_ascii=False)
    return ann_rgb, formatted_json_str


# Create Gradio Blocks Interface
with gr.Blocks(title="VisionOps Guard - Safety PPE AI", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🛡️ VisionOps Guard — Real-Time Safety PPE Inspection")
    gr.Markdown(
        "Sistem deteksi visual Alat Pelindung Diri (APD/PPE) dan K3 industri berbasis ONNX Runtime. "
        "Pilih tab **Kamera Langsung** atau **Upload File Gambar** di bawah untuk memulai inspeksi."
    )

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Tabs():
                with gr.TabItem("📸 Kamera Langsung (Webcam)"):
                    webcam_img = gr.Image(
                        sources=["webcam"],
                        type="numpy",
                        label="Live Webcam Stream"
                    )
                    gr.Markdown("💡 *Posisikan diri Anda, klik icon kamera **[📷]** di tengah bawah video untuk snap foto, lalu klik tombol periksa di bawah:*")
                    webcam_btn = gr.Button("⚡ Periksa Kepatuhan K3 dari Kamera", variant="primary")

                with gr.TabItem("📁 Upload File Gambar"):
                    upload_img = gr.Image(
                        sources=["upload"],
                        type="numpy",
                        label="Pilih atau Drag-and-Drop Gambar Proyek"
                    )
                    upload_btn = gr.Button("⚡ Periksa Kepatuhan K3 dari File", variant="primary")

            conf_slider = gr.Slider(
                minimum=0.05,
                maximum=1.0,
                value=0.20,
                step=0.05,
                label="Confidence Threshold (Rekomendasi: 0.15 - 0.25)"
            )
        
        with gr.Column(scale=1):
            output_img = gr.Image(type="numpy", label="ONNX Detection Result")
            output_details = gr.Textbox(label="Compliance Analytics & Bounding Boxes", lines=14)

    # Event Handlers for Webcam
    webcam_btn.click(
        fn=inspect_safety_ppe,
        inputs=[webcam_img, conf_slider],
        outputs=[output_img, output_details],
        api_name=False
    )
    webcam_img.change(
        fn=inspect_safety_ppe,
        inputs=[webcam_img, conf_slider],
        outputs=[output_img, output_details],
        api_name=False
    )

    # Event Handlers for File Upload
    upload_btn.click(
        fn=inspect_safety_ppe,
        inputs=[upload_img, conf_slider],
        outputs=[output_img, output_details],
        api_name=False
    )
    upload_img.change(
        fn=inspect_safety_ppe,
        inputs=[upload_img, conf_slider],
        outputs=[output_img, output_details],
        api_name=False
    )

    # Pre-loaded Construction & PPE Test Examples
    sample_files = [
        ["data/raw/images/test/ppe_test_0000.jpg", 0.20],
        ["data/raw/images/test/ppe_test_0001.jpg", 0.20],
        ["data/raw/images/test/ppe_test_0002.jpg", 0.20],
        ["data/raw/images/test/ppe_test_0003.jpg", 0.20],
        ["data/raw/images/test/ppe_test_0004.jpg", 0.20],
    ]
    existing_samples = [s for s in sample_files if Path(s[0]).exists()]
    if existing_samples:
        gr.Examples(
            examples=existing_samples,
            inputs=[upload_img, conf_slider],
            label="📁 Klik Contoh Gambar Proyek Lapangan (Pre-loaded Test Images)"
        )


if __name__ == "__main__":
    demo.launch()
