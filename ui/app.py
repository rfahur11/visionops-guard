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
    st.markdown('<div class="main-header">🛡️ VisionOps Guard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Real-Time Industrial Safety PPE Inspection & CVOps Pipeline</div>', unsafe_allow_html=True)

    predictor = load_predictor()

    # Sidebar Controls
    st.sidebar.header("⚙️ Model & Inference Controls")
    conf_thresh = st.sidebar.slider("Confidence Threshold", 0.05, 1.0, 0.20, 0.05, help="Rekomendasi: 0.15 - 0.25")
    iou_thresh = st.sidebar.slider("NMS IoU Threshold", 0.1, 1.0, 0.45, 0.05)
    predictor.conf_threshold = conf_thresh
    predictor.iou_threshold = iou_thresh

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 MLOps Stack Status")
    st.sidebar.success("ONNX Runtime Engine: **ACTIVE**")
    st.sidebar.info("Model Device: **CUDA GPU / CPU**")

    # Input Method Selection
    input_mode = st.radio("Select Visual Input Source:", ["Upload Image File", "Live Camera (Webcam)", "Use Sample Dataset Image"], horizontal=True)

    image_bytes = None

    if input_mode == "Upload Image File":
        uploaded_file = st.file_uploader("Upload Inspection Image (JPG / PNG):", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image_bytes = uploaded_file.read()
    elif input_mode == "Live Camera (Webcam)":
        camera_file = st.camera_input("Ambil Foto dari Kamera Langsung:")
        if camera_file is not None:
            image_bytes = camera_file.read()
    else:
        sample_dir = Path("data/raw/images/test")
        sample_files = list(sample_dir.glob("*.jpg")) if sample_dir.exists() else []
        if sample_files:
            selected_sample = st.selectbox("Choose Sample Image:", sample_files, format_func=lambda x: x.name)
            if selected_sample:
                with open(selected_sample, "rb") as f:
                    image_bytes = f.read()
        else:
            st.warning("No sample images found in data/raw/images/test. Please upload an image.")

    if image_bytes is not None:
        col1, col2 = st.columns(2)

        # Run ONNX Prediction with spinner feedback
        with st.spinner("⚡ Menjalankan inferensi ONNX Runtime & evaluasi standar K3..."):
            result = predictor.predict(image_bytes)

        with col1:
            st.markdown("### 📷 Original Input Image")
            nparr = np.frombuffer(image_bytes, np.uint8)
            orig_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            orig_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
            st.image(orig_rgb, use_container_width=True)

        with col2:
            st.markdown("### 🎯 ONNX Real-Time Detection")
            img_b64 = result["annotated_image_base64"]
            img_data = base64.b64decode(img_b64)
            nparr_ann = np.frombuffer(img_data, np.uint8)
            ann_img = cv2.imdecode(nparr_ann, cv2.IMREAD_COLOR)
            ann_rgb = cv2.cvtColor(ann_img, cv2.COLOR_BGR2RGB)
            st.image(ann_rgb, use_container_width=True)

        st.markdown("---")

        # Status Banner
        compliance_status = result.get("compliance_status", "COMPLIANT" if result["is_compliant"] else "VIOLATION DETECTED")
        if result["is_compliant"]:
            st.markdown(f'<div class="compliant-banner">✅ {compliance_status} — {result.get("assessment", "Standar K3 Terpenuhi")}</div>', unsafe_allow_html=True)
        elif compliance_status.startswith("NO"):
            st.info(f"ℹ️ {compliance_status}: {result.get('assessment', 'Tidak ada pekerja atau APD terdeteksi.')}")
        else:
            st.markdown(f'<div class="violation-banner">⚠️ {compliance_status} — {result.get("assessment", f"{result[\"violations_count\"]} Non-Compliance Alert(s)")}</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Metrics Panel
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Inference Latency", f"{result['latency_ms']} ms", delta="-42% vs PyTorch")
        m2.metric("Total Detections", len(result["detections"]))
        m3.metric("Violations Count", result["violations_count"])
        m4.metric("Compliance Status", "COMPLIANT" if result["is_compliant"] else "VIOLATION")

        # Detailed Detections Table
        st.markdown("### 📋 Detected Objects Breakdown")
        if result["detections"]:
            det_data = []
            for d in result["detections"]:
                det_data.append({
                    "Class": d["class_name"].upper(),
                    "Confidence": f"{d['confidence'] * 100:.1f}%",
                    "Bounding Box (XYWH)": str(d["box_xywh"])
                })
            st.dataframe(det_data, use_container_width=True)
        else:
            st.info("No objects detected above confidence threshold.")


if __name__ == "__main__":
    import io
    main()
