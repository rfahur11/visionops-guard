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
def inspect_safety_ppe(input_image: np.ndarray, conf_threshold: float = 0.20, lang_choice: str = "Bahasa Indonesia (ID)", progress=gr.Progress(track_tqdm=True)):
    """
    Gradio prediction function with bilingual support and real-time animated progress bar.
    """
    is_en = "EN" in str(lang_choice)
    if input_image is None:
        msg = "⚠️ No image provided. Please upload an image or click the camera icon to snap a photo." if is_en else "⚠️ Belum ada foto yang dipilih. Silakan upload gambar atau klik tombol kamera untuk menjepret foto."
        return None, msg

    prep_msg = "📸 Preparing and decoding image frame..." if is_en else "📸 Mempersiapkan frame gambar..."
    progress(0.15, desc=prep_msg)

    # Convert RGB to BGR for OpenCV encoding
    img_bgr = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".jpg", img_bgr)
    img_bytes = buffer.tobytes()

    # Update confidence threshold
    predictor.conf_threshold = conf_threshold

    infer_msg = "⚡ Running ONNX Runtime inference acceleration..." if is_en else "⚡ Menjalankan inferensi ONNX Runtime..."
    progress(0.50, desc=infer_msg)
    # Run Prediction with selected language
    result = predictor.predict(img_bytes, lang="en" if is_en else "id")

    eval_msg = "🛡️ Analyzing safety PPE compliance & bounding boxes..." if is_en else "🛡️ Menganalisis kepatuhan standar K3 APD..."
    progress(0.85, desc=eval_msg)
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
        ("safety_assessment" if is_en else "k3_assessment"): result.get("assessment", ""),
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

    done_msg = "✅ Inspection Completed!" if is_en else "✅ Analisis Selesai!"
    progress(1.0, desc=done_msg)
    formatted_json_str = json.dumps(summary, indent=2, ensure_ascii=False)
    return ann_rgb, formatted_json_str


# Create Gradio Blocks Interface
with gr.Blocks(title="VisionOps Guard - Safety PPE AI", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🛡️ VisionOps Guard — Real-Time Safety PPE Inspection")
    gr.Markdown(
        "Sistem deteksi visual Alat Pelindung Diri (APD/PPE) dan K3 industri berbasis ONNX Runtime. "
        "Supports Real-Time Industrial PPE Inspection & Multi-Language Display."
    )

    with gr.Row():
        lang_radio = gr.Radio(
            choices=["Bahasa Indonesia (ID)", "English (EN)"],
            value="Bahasa Indonesia (ID)",
            label="🌐 Bahasa / Language",
            interactive=True
        )

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Tabs():
                with gr.TabItem("📸 Kamera Langsung / Live Webcam"):
                    webcam_img = gr.Image(
                        sources=["webcam"],
                        type="numpy",
                        label="Live Webcam Stream"
                    )
                    gr.Markdown("💡 *Posisikan diri Anda, klik icon kamera **[📷]** di tengah bawah video untuk snap foto, lalu klik tombol periksa:*")
                    webcam_btn = gr.Button("⚡ Periksa Kepatuhan / Inspect Safety", variant="primary")

                with gr.TabItem("📁 Upload File Gambar / Upload Image"):
                    upload_img = gr.Image(
                        sources=["upload"],
                        type="numpy",
                        label="Pilih atau Drag-and-Drop Gambar Proyek"
                    )
                    upload_btn = gr.Button("⚡ Periksa Kepatuhan / Inspect Safety", variant="primary")

            conf_slider = gr.Slider(
                minimum=0.05,
                maximum=1.0,
                value=0.20,
                step=0.05,
                label="Confidence Threshold (Rekomendasi / Recommended: 0.15 - 0.25)"
            )
        
        with gr.Column(scale=1):
            output_img = gr.Image(type="numpy", label="ONNX Detection Result")
            output_details = gr.Textbox(label="Compliance Analytics & Bounding Boxes", lines=14)

    # Event Handlers for Webcam
    webcam_btn.click(
        fn=inspect_safety_ppe,
        inputs=[webcam_img, conf_slider, lang_radio],
        outputs=[output_img, output_details],
        api_name=False
    )
    webcam_img.change(
        fn=inspect_safety_ppe,
        inputs=[webcam_img, conf_slider, lang_radio],
        outputs=[output_img, output_details],
        api_name=False
    )

    # Event Handlers for File Upload
    upload_btn.click(
        fn=inspect_safety_ppe,
        inputs=[upload_img, conf_slider, lang_radio],
        outputs=[output_img, output_details],
        api_name=False
    )
    upload_img.change(
        fn=inspect_safety_ppe,
        inputs=[upload_img, conf_slider, lang_radio],
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
