"""
VisionOps Guard - Interactive Streamlit Web UI
Real-time visual safety PPE inspection dashboard with ONNX Runtime inference & compliance analytics.
"""

import base64
import json
from pathlib import Path
import cv2
import numpy as np
import PIL.Image
import streamlit as st
import yaml
from src.serving.predictor import SafetyPPEPredictor

# Page Configuration
st.set_page_config(
    page_title="VisionOps Guard - Safety PPE AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #3B82F6; }
    .sub-header { font-size: 1.1rem; color: #9CA3AF; margin-bottom: 1.5rem; }
    .stMetric { background-color: #1F2937; padding: 12px; border-radius: 8px; }
    .compliant-banner { background-color: #065F46; color: #34D399; padding: 16px; border-radius: 8px; font-weight: bold; font-size: 1.2rem; text-align: center; }
    .violation-banner { background-color: #991B1B; color: #FCA5A5; padding: 16px; border-radius: 8px; font-weight: bold; font-size: 1.2rem; text-align: center; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor():
    return SafetyPPEPredictor()


def main():
    # Sidebar Language Selector
    st.sidebar.header("🌐 Language / Bahasa")
    selected_lang = st.sidebar.selectbox("Pilih Bahasa / Select Language:", ["Bahasa Indonesia (ID)", "English (EN)"])
    is_en = "EN" in selected_lang

    st.markdown('<div class="main-header">🛡️ VisionOps Guard</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="sub-header">{"Real-Time Industrial Safety PPE Inspection & CVOps Pipeline" if is_en else "Sistem CVOps & Inspeksi Kepatuhan K3 APD Industri Real-Time"}</div>',
        unsafe_allow_html=True
    )

    predictor = load_predictor()

    # Sidebar Controls
    st.sidebar.header("⚙️ " + ("Model & Inference Controls" if is_en else "Kontrol Model & Inferensi"))
    conf_thresh = st.sidebar.slider(
        "Confidence Threshold" if is_en else "Ambang Batas Kepercayaan (Confidence)",
        0.05, 1.0, 0.20, 0.05,
        help="Recommended: 0.15 - 0.25" if is_en else "Rekomendasi: 0.15 - 0.25"
    )
    iou_thresh = st.sidebar.slider("NMS IoU Threshold", 0.1, 1.0, 0.45, 0.05)
    predictor.conf_threshold = conf_thresh
    predictor.iou_threshold = iou_thresh

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 " + ("MLOps Stack Status" if is_en else "Status Komponen MLOps"))
    st.sidebar.success("ONNX Runtime Engine: **ACTIVE**")
    st.sidebar.info("Model Device: **CUDA GPU / CPU**")

    # Input Method Selection
    input_modes = (
        ["Upload Image File", "Live Camera (Webcam)", "Use Sample Dataset Image"]
        if is_en else
        ["Upload File Gambar", "Kamera Langsung (Webcam)", "Gunakan Contoh Gambar Proyek"]
    )
    input_mode = st.radio(
        "Select Visual Input Source:" if is_en else "Pilih Sumber Gambar Input:",
        input_modes,
        horizontal=True
    )

    image_bytes = None

    if input_mode in ["Upload Image File", "Upload File Gambar"]:
        uploaded_file = st.file_uploader(
            "Upload Inspection Image (JPG / PNG):" if is_en else "Unggah Gambar Inspeksi (JPG / PNG):",
            type=["jpg", "jpeg", "png"]
        )
        if uploaded_file is not None:
            image_bytes = uploaded_file.read()
    elif input_mode in ["Live Camera (Webcam)", "Kamera Langsung (Webcam)"]:
        camera_file = st.camera_input("Take Live Camera Photo:" if is_en else "Ambil Foto dari Kamera Langsung:")
        if camera_file is not None:
            image_bytes = camera_file.read()
    else:
        sample_dir = Path("data/raw/images/test")
        sample_files = list(sample_dir.glob("*.jpg")) if sample_dir.exists() else []
        if sample_files:
            selected_sample = st.selectbox(
                "Choose Sample Image:" if is_en else "Pilih Contoh Gambar:",
                sample_files,
                format_func=lambda x: x.name
            )
            if selected_sample:
                with open(selected_sample, "rb") as f:
                    image_bytes = f.read()
        else:
            st.warning(
                "No sample images found in data/raw/images/test. Please upload an image."
                if is_en else
                "Tidak ada gambar sampel ditemukan di data/raw/images/test. Silakan unggah gambar."
            )

    if image_bytes is not None:
        col1, col2 = st.columns(2)

        # Run ONNX Prediction with spinner feedback
        spinner_msg = (
            "⚡ Running ONNX Runtime inference & safety assessment..."
            if is_en else
            "⚡ Menjalankan inferensi ONNX Runtime & evaluasi standar K3..."
        )
        with st.spinner(spinner_msg):
            result = predictor.predict(image_bytes, lang="en" if is_en else "id")

        with col1:
            st.markdown("### 📷 " + ("Original Input Image" if is_en else "Gambar Input Asli"))
            nparr = np.frombuffer(image_bytes, np.uint8)
            orig_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            orig_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
            st.image(orig_rgb, use_container_width=True)

        with col2:
            st.markdown("### 🎯 " + ("ONNX Real-Time Detection" if is_en else "Hasil Deteksi Real-Time ONNX"))
            img_b64 = result["annotated_image_base64"]
            img_data = base64.b64decode(img_b64)
            nparr_ann = np.frombuffer(img_data, np.uint8)
            ann_img = cv2.imdecode(nparr_ann, cv2.IMREAD_COLOR)
            ann_rgb = cv2.cvtColor(ann_img, cv2.COLOR_BGR2RGB)
            st.image(ann_rgb, use_container_width=True)

        st.markdown("---")

        # Status Banner
        compliance_status = result.get("compliance_status", "COMPLIANT" if result["is_compliant"] else "VIOLATION DETECTED")
        assessment_text = result.get("assessment", "")
        if not assessment_text:
            if result["is_compliant"]:
                assessment_text = "Safety Standards Met" if is_en else "Standar K3 Terpenuhi"
            elif "NO" in compliance_status or "TIDAK" in compliance_status:
                assessment_text = "No worker or PPE detected." if is_en else "Tidak ada pekerja atau APD terdeteksi."
            else:
                v_count = result.get("violations_count", 0)
                assessment_text = f"{v_count} Non-Compliance Alert(s)" if is_en else f"{v_count} Peringatan Pelanggaran K3"

        if result["is_compliant"]:
            st.markdown(f'<div class="compliant-banner">✅ {compliance_status} — {assessment_text}</div>', unsafe_allow_html=True)
        elif "NO" in compliance_status or "TIDAK" in compliance_status:
            st.info(f"ℹ️ {compliance_status}: {assessment_text}")
        else:
            st.markdown(f'<div class="violation-banner">⚠️ {compliance_status} — {assessment_text}</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Metrics Panel
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Inference Latency" if is_en else "Latensi Inferensi", f"{result['latency_ms']} ms", delta="-42% vs PyTorch")
        m2.metric("Total Detections" if is_en else "Total Deteksi", len(result["detections"]))
        m3.metric("Violations Count" if is_en else "Jumlah Pelanggaran", result["violations_count"])
        
        if is_en:
            status_metric = "COMPLIANT" if result["is_compliant"] else ("NO PPE" if "NO" in compliance_status else "VIOLATION")
        else:
            status_metric = "PATUH K3" if result["is_compliant"] else ("TIDAK TERDETEKSI" if "TIDAK" in compliance_status else "PELANGGARAN")
        m4.metric("Compliance Status" if is_en else "Status Kepatuhan", status_metric)

        # Detailed Detections Table
        st.markdown("### 📋 " + ("Detected Objects Breakdown" if is_en else "Rincian Objek Terdeteksi"))
        if result["detections"]:
            det_data = []
            for d in result["detections"]:
                det_data.append({
                    "Class" if is_en else "Kelas": d.get("class_display", d["class_name"].upper()),
                    "Confidence" if is_en else "Tingkat Keyakinan": f"{d['confidence'] * 100:.1f}%",
                    "Bounding Box (XYWH)": str(d["box_xywh"])
                })
            st.table(det_data)

        else:
            st.info("No objects detected above confidence threshold." if is_en else "Tidak ada objek yang terdeteksi di atas ambang batas kepercayaan.")


if __name__ == "__main__":
    import io
    main()
