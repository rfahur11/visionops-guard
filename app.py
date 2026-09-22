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

# Optional ZeroGPU support if Space has GPU assigned (dynamic import prevents IDE linter errors on local machine)
try:
    import importlib
    spaces_mod = importlib.import_module("spaces")
    gpu_decorator = getattr(spaces_mod, "GPU", lambda fn: fn)
except (ImportError, ModuleNotFoundError, Exception):
    def gpu_decorator(fn):
        return fn

# Initialize Predictor Singleton
predictor = SafetyPPEPredictor()


def create_placeholder_image(lang_choice: str = "Bahasa Indonesia (ID)") -> np.ndarray:
    """Generates a dark slate graphic placeholder image when no camera snapshot is present."""
    is_en = "EN" in str(lang_choice)
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (31, 41, 55)  # Dark slate #1F2937

    title = "NO WEBCAM PHOTO CAPTURED" if is_en else "BELUM ADA FOTO DARI KAMERA"
    line1 = "Please click the camera icon [📷] on the video" if is_en else "Silakan klik icon camera [📷] pada video"
    line2 = "to snap a photo first, then click Inspect Safety." if is_en else "untuk menjepret foto sebelum memeriksa."

    cv2.putText(img, title, (45, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (239, 68, 68), 2, cv2.LINE_AA)
    cv2.putText(img, line1, (55, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (229, 231, 235), 2, cv2.LINE_AA)
    cv2.putText(img, line2, (55, 305), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (229, 231, 235), 2, cv2.LINE_AA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


@gpu_decorator
def inspect_safety_ppe(input_image: np.ndarray, conf_threshold: float = 0.20, lang_choice: str = "Bahasa Indonesia (ID)", progress=gr.Progress(track_tqdm=True)):
    """
    Gradio generator function with bilingual support, instant preview yield, and zero-blank progress feedback.
    """
    is_en = "EN" in str(lang_choice)

    if input_image is None:
        empty_res = {
            "status": "AWAITING SNAPSHOT ℹ️" if is_en else "MENUNGGU SNAPSHOT ℹ️",
            ("safety_assessment" if is_en else "evaluasi_k3"): (
                "No camera photo snapped. Please click the camera icon [📷] on the video feed to snap a photo."
                if is_en else
                "Belum ada foto yang dijepret. Silakan klik icon kamera [📷] pada video untuk mengunggah/snap foto."
            ),
            "inference_latency_ms": 0.0,
            ("detections_count" if is_en else "total_deteksi"): 0
        }
        placeholder_img = create_placeholder_image(lang_choice)
        yield placeholder_img, json.dumps(empty_res, indent=2, ensure_ascii=False)
        return

    # Step 1: Immediately yield intermediate processing status to eliminate any blank delay!
    processing_json = json.dumps({
        "status": "⏳ PROCESSING INFERENCE..." if is_en else "⏳ SEDANG MEMPROSES...",
        ("safety_assessment" if is_en else "evaluasi_k3"): (
            "⚡ Running ONNX Runtime inference engine & safety evaluation..."
            if is_en else
            "⚡ Menjalankan model ONNX Runtime & evaluasi kepatuhan K3..."
        ),
        "inference_latency_ms": "Calculating...",
        ("detections_count" if is_en else "total_deteksi"): 0
    }, indent=2, ensure_ascii=False)

    prep_msg = "📸 Preparing and decoding image frame..." if is_en else "📸 Mempersiapkan frame gambar..."
    progress(0.15, desc=prep_msg)
    yield input_image, processing_json

    # Step 2: Convert RGB to BGR and encode
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

    status_text = result.get("compliance_status", "COMPLIANT" if is_en else "PATUH STANDAR K3")
    if result["is_compliant"]:
        status_display = f"{status_text} ✅"
    elif "NO" in status_text or "TIDAK" in status_text:
        status_display = f"{status_text} ℹ️"
    else:
        status_display = f"{status_text} ⚠️"

    if is_en:
        summary = {
            "status": status_display,
            "safety_assessment": result.get("assessment", ""),
            "violations_count": result["violations_count"],
            "violation_details": result.get("violation_details", []),
            "inference_latency_ms": round(result["latency_ms"], 2),
            "detections_count": len(result["detections"]),
            "detected_objects": [
                {
                    "class": d["class_name"].upper(),
                    "description": d.get("class_display", d["class_name"].upper()),
                    "confidence": f"{d['confidence']*100:.1f}%",
                    "box_xyxy": d["box_xyxy"]
                }
                for d in result["detections"]
            ]
        }
    else:
        summary = {
            "status": status_display,
            "evaluasi_k3": result.get("assessment", ""),
            "jumlah_pelanggaran": result["violations_count"],
            "rincian_pelanggaran": result.get("violation_details", []),
            "latensi_inferensi_ms": round(result["latency_ms"], 2),
            "total_deteksi": len(result["detections"]),
            "objek_terdeteksi": [
                {
                    "kelas": d.get("class_display", d["class_name"].upper()),
                    "tingkat_keyakinan": f"{d['confidence']*100:.1f}%",
                    "koordinat_kotak": d["box_xyxy"]
                }
                for d in result["detections"]
            ]
        }

    done_msg = "✅ Inspection Completed!" if is_en else "✅ Analisis Selesai!"
    progress(1.0, desc=done_msg)
    formatted_json_str = json.dumps(summary, indent=2, ensure_ascii=False)
    yield ann_rgb, formatted_json_str


def switch_language(lang_choice, webcam_image, upload_image, conf_threshold):
    """
    Dynamically updates all UI text elements when the user toggles language.
    If an image is currently loaded, it immediately re-evaluates the output.
    """
    is_en = "EN" in str(lang_choice)
    desc_val = (
        "Industrial Safety PPE visual inspection and compliance monitoring system powered by ONNX Runtime."
        if is_en else
        "Sistem deteksi visual Alat Pelindung Diri (APD/PPE) dan kepatuhan K3 industri berbasis ONNX Runtime."
    )
    cam_hint_val = (
        "💡 *Position yourself in front of the camera, click the camera icon **[📷]** on the video to snap a photo, then click inspect:* "
        if is_en else
        "💡 *Posisikan diri Anda di depan kamera, klik icon kamera **[📷]** di tengah video untuk snap foto, lalu klik tombol periksa:*"
    )
    webcam_btn_val = "⚡ Inspect Safety Compliance" if is_en else "⚡ Periksa Kepatuhan K3"
    upload_btn_val = "⚡ Inspect Safety Compliance" if is_en else "⚡ Periksa Kepatuhan K3"
    slider_label_val = (
        "Confidence Threshold (Recommended: 0.15 - 0.25)"
        if is_en else
        "Ambang Batas Kepercayaan (Rekomendasi: 0.15 - 0.25)"
    )
    webcam_label_val = "Live Webcam Stream" if is_en else "Tangkapan Kamera Langsung (Webcam)"
    upload_label_val = "Upload Inspection Image (JPG / PNG)" if is_en else "Unggah Gambar Inspeksi (JPG / PNG)"
    out_img_label_val = "ONNX Detection Visualization" if is_en else "Hasil Deteksi Visual ONNX"
    out_details_label_val = "Safety Compliance Analytics & Bounding Boxes" if is_en else "Analisis Kepatuhan K3 & Bounding Box"
    tab_cam_val = "📸 Live Camera (Webcam)" if is_en else "📸 Kamera Langsung (Webcam)"
    tab_up_val = "📁 Upload Image File" if is_en else "📁 Unggah File Gambar"

    active_img = webcam_image if webcam_image is not None else upload_image
    if active_img is not None:
        # Run generator to get final output
        res_list = list(inspect_safety_ppe(active_img, conf_threshold, lang_choice))
        ann_rgb, json_str = res_list[-1]
        return (
            gr.update(value=desc_val),
            gr.update(value=cam_hint_val),
            gr.update(value=webcam_btn_val),
            gr.update(value=upload_btn_val),
            gr.update(label=slider_label_val),
            gr.update(label=webcam_label_val),
            gr.update(label=upload_label_val),
            gr.update(label=tab_cam_val),
            gr.update(label=tab_up_val),
            gr.update(label=out_img_label_val, value=ann_rgb),
            gr.update(label=out_details_label_val, value=json_str)
        )
    else:
        init_msg = (
            json.dumps({
                "status": "AWAITING INPUT ℹ️",
                "safety_assessment": "Please snap a webcam photo or upload an image to start inspection.",
                "violations_count": 0,
                "inference_latency_ms": 0.0,
                "detections_count": 0
            }, indent=2)
            if is_en else
            json.dumps({
                "status": "MENUNGGU INPUT ℹ️",
                "evaluasi_k3": "Silakan ambil foto webcam atau unggah gambar untuk memulai inspeksi K3.",
                "jumlah_pelanggaran": 0,
                "latensi_inferensi_ms": 0.0,
                "total_deteksi": 0
            }, indent=2, ensure_ascii=False)
        )
        placeholder_img = create_placeholder_image(lang_choice)
        return (
            gr.update(value=desc_val),
            gr.update(value=cam_hint_val),
            gr.update(value=webcam_btn_val),
            gr.update(value=upload_btn_val),
            gr.update(label=slider_label_val),
            gr.update(label=webcam_label_val),
            gr.update(label=upload_label_val),
            gr.update(label=tab_cam_val),
            gr.update(label=tab_up_val),
            gr.update(label=out_img_label_val, value=placeholder_img),
            gr.update(label=out_details_label_val, value=init_msg)
        )


# Initial Indonesian placeholder string
initial_details_placeholder = json.dumps({
    "status": "MENUNGGU INPUT ℹ️",
    "evaluasi_k3": "Silakan ambil foto webcam atau unggah gambar untuk memulai inspeksi K3.",
    "jumlah_pelanggaran": 0,
    "latensi_inferensi_ms": 0.0,
    "total_deteksi": 0
}, indent=2, ensure_ascii=False)

# Create Gradio Blocks Interface
with gr.Blocks(title="VisionOps Guard - Safety PPE AI", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🛡️ VisionOps Guard — Real-Time Safety PPE Inspection")
    desc_md = gr.Markdown(
        "Sistem deteksi visual Alat Pelindung Diri (APD/PPE) dan kepatuhan K3 industri berbasis ONNX Runtime."
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
                with gr.TabItem("📸 Kamera Langsung (Webcam)") as tab_cam:
                    webcam_img = gr.Image(
                        sources=["webcam"],
                        type="numpy",
                        label="Tangkapan Kamera Langsung (Webcam)"
                    )
                    cam_hint = gr.Markdown("💡 *Posisikan diri Anda di depan kamera, klik icon kamera **[📷]** di tengah video untuk snap foto, lalu klik tombol periksa:*")
                    webcam_btn = gr.Button("⚡ Periksa Kepatuhan K3", variant="primary")

                with gr.TabItem("📁 Unggah File Gambar") as tab_up:
                    upload_img = gr.Image(
                        sources=["upload"],
                        type="numpy",
                        label="Unggah Gambar Inspeksi (JPG / PNG)"
                    )
                    upload_btn = gr.Button("⚡ Periksa Kepatuhan K3", variant="primary")

            conf_slider = gr.Slider(
                minimum=0.05,
                maximum=1.0,
                value=0.20,
                step=0.05,
                label="Ambang Batas Kepercayaan (Rekomendasi: 0.15 - 0.25)"
            )
        
        with gr.Column(scale=1):
            output_img = gr.Image(
                type="numpy",
                label="Hasil Deteksi Visual ONNX",
                value=create_placeholder_image("Bahasa Indonesia (ID)")
            )
            output_details = gr.Textbox(
                label="Analisis Kepatuhan K3 & Bounding Box",
                value=initial_details_placeholder,
                lines=14
            )

    # Event Handlers for Language Switching
    lang_radio.change(
        fn=switch_language,
        inputs=[lang_radio, webcam_img, upload_img, conf_slider],
        outputs=[
            desc_md,
            cam_hint,
            webcam_btn,
            upload_btn,
            conf_slider,
            webcam_img,
            upload_img,
            tab_cam,
            tab_up,
            output_img,
            output_details
        ],
        show_progress="full",
        api_name=False
    )

    # Event Handlers for Webcam
    webcam_btn.click(
        fn=inspect_safety_ppe,
        inputs=[webcam_img, conf_slider, lang_radio],
        outputs=[output_img, output_details],
        show_progress="full",
        api_name=False
    )
    webcam_img.change(
        fn=inspect_safety_ppe,
        inputs=[webcam_img, conf_slider, lang_radio],
        outputs=[output_img, output_details],
        show_progress="full",
        api_name=False
    )

    # Event Handlers for File Upload
    upload_btn.click(
        fn=inspect_safety_ppe,
        inputs=[upload_img, conf_slider, lang_radio],
        outputs=[output_img, output_details],
        show_progress="full",
        api_name=False
    )
    upload_img.change(
        fn=inspect_safety_ppe,
        inputs=[upload_img, conf_slider, lang_radio],
        outputs=[output_img, output_details],
        show_progress="full",
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
            label="📁 Contoh Gambar Proyek Lapangan (Pre-loaded Test Images)"
        )


if __name__ == "__main__":
    demo.launch()
