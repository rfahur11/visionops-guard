# SYSTEM STATE - VisionOps Guard

**Last Synchronized:** 2026-09-23 23:05:00 (WIB)  
**System Version:** v1.3.0  
**Environment:** Windows 11 (Host GPU RTX 4060) + Docker Desktop (WSL2 Linux Containers)  
**Deployment Target:** GitHub & Hugging Face Spaces (`https://huggingface.co/spaces/rfahrur6045/mlops-final-copy`)

---

## 📌 Status Ringkasan Proyek

VisionOps Guard adalah sistem MLOps end-to-end berbasis Computer Vision untuk deteksi Alat Pelindung Diri (K3 / PPE) secara real-time. Sistem telah mengadopsi arsitektur state-of-the-art **Ultralytics YOLO26** (`yolo26n`), dikontainerisasi penuh menggunakan Docker Compose 5-service (FastAPI, Streamlit, Prometheus, Grafana, MLflow), didukung oleh Observability Stack & Model Management MCP Servers, serta otomatisasi pipeline CI/CD GitHub Actions ke Hugging Face Spaces.

---

## 🛠️ Komponen Sistem & Status Terbaru

| Komponen | Status | Deskripsi & Port |
| :--- | :--- | :--- |
| **YOLO26 ONNX Engine** | 🟢 Active | Model Champion **YOLO26n** Quantized FP16 (`models/visionops_guard.onnx`, 4.9MB) berlatensi ~1.3ms. Evaluasi: **Precision 91.5%**, **Recall 67.0%**, **mAP50 72.6%**. |
| **FastAPI REST Service** | 🟢 Active | Microservice di port **8000** (`src/serving/main:app`) dengan endpoint `/predict`, `/health`, dan Prometheus `/metrics`. |
| **Streamlit Web UI** | 🟢 Active | Web dashboard interaktif di port **8501** (`ui/app.py`), menggunakan `st.table()` dan input preview real-time. |
| **MLflow Experiment Tracking** | 🟢 Active | Container UI pelacakan eksperimen & registry model di port **5000** (`visionops-mlflow`) membaca volume `./mlruns`. |
| **Prometheus Monitoring** | 🟢 Active | Observabilitas matriks time-series di port **9090** (`config/prometheus.yml`) melakukan scraping live dari `visionops-api:8000`. |
| **Grafana Visual Dashboard** | 🟢 Active | Dashboard analitik visual eksekutif di port **3000** (`visionops-grafana`) dengan auto-provisioned Prometheus datasource & pre-built dashboard (*Executive Command Center*). |
| **Grafana & MLflow MCP** | 🟢 Active | Model Context Protocol integration (`@grafana/mcp-server` & `mlflow-mcp-server`) di `mcp_config.json` via token dan stdio. |
| **Automated CI/CD Workflow** | 🟢 Active | GitHub Actions (`.github/workflows/ci_cd.yaml`) otomatis menjalankan Data QA, Pytest unit tests, dan auto-deploy ke Hugging Face Spaces. |

---

## 🔄 Pembaruan Terakhir (Recent Updates)

1. **Upgrade Arsitektur ke Ultralytics YOLO26 (`yolo26n`):**
   - Mengadopsi arsitektur terbaru **YOLO26** (120 layers, 2.37M parameters, 5.3 GFLOPs).
   - Melakukan retraining dengan hyperparameter tuning: **Optimizer AdamW**, initial lr `0.005`, weight decay `0.0005`, 15 epochs pada GPU native RTX 4060.
   - Hasil benchmark validasi:
     - **mAP50**: `0.7264` (72.6%)
     - **Precision**: `0.9147` (91.5%)
     - **Recall**: `0.6700` (67.0%)
     - **Class Person**: mAP50 `98.6%`, Precision `95.0%`
     - **Class Vest**: mAP50 `87.8%`, Precision `88.9%`
     - **Class Boots**: mAP50 `81.9%`, Precision `97.5%`
     - **Class Gloves**: mAP50 `78.7%`, Precision `90.4%`
   - Auto-ekspor model champion ke format ONNX FP16 (`models/visionops_guard.onnx`, 4.9 MB) dengan `opset 18` dan `onnxslim`.
2. **Implementasi MLflow Tracking & Containerization:**
   - Ditambahkan container `mlflow` di `docker-compose.yml` pada port **5000** dengan bind mount `./mlruns:/app/mlruns` dan env `MLFLOW_ALLOW_FILE_STORE=true`.
   - Menghubungkan training script `src/models/train.py` untuk streaming metrics dan parameter ke tracking URI `http://localhost:5000`.
   - Mengkonfigurasi MLflow MCP Server di `C:\Users\black\.gemini\config\mcp_config.json` serta membuat agent skill `.agents/skills/mlflow-model-management/SKILL.md`.
3. **Implementasi Full Observability Stack (Grafana + Prometheus):**
   - Menjalankan Grafana di port **3000** dengan auto-provisioned datasource Prometheus di port **9090**.
   - Menyediakan Dashboard-as-Code otomatis (*VisionOps Guard - Executive Command Center*) di `config/grafana/provisioning/dashboards/visionops_guard.json`.
   - Mengintegrasikan `@grafana/mcp-server` dengan token Service Account.
4. **Automated CI/CD Pipeline (GitHub Actions ke HF Spaces):**
   - Menggabungkan pipeline pengujian dan deployment ke dalam `.github/workflows/ci_cd.yaml`.
   - Pipeline mencakup tahap verifikasi data QA, unit test (Pytest 6/6 lulus), dan sinkronisasi otomatis ke remote Hugging Face Spaces (`https://huggingface.co/spaces/rfahrur6045/mlops-final-copy`) menggunakan GitHub secret `HF_TOKEN`.

---

## 🚀 Panduan Eksekusi Sistem

### A. Menjalankan Seluruh Ekosistem (Docker Compose)
```powershell
docker compose up -d
```
- 🎨 **Streamlit Web UI:** `http://localhost:8501`
- ⚡ **FastAPI REST Docs:** `http://localhost:8000/docs`
- 🧪 **MLflow Model Registry:** `http://localhost:5000`
- 📊 **Prometheus Engine:** `http://localhost:9090`
- 📈 **Grafana Command Center:** `http://localhost:3000` *(Login: admin / admin)*

### B. Menjalankan Retraining Model YOLO26 (Lokal GPU)
```powershell
.venv\Scripts\python.exe src/models/train.py
```

### C. Menjalankan Unit Testing
```powershell
.venv\Scripts\python.exe -m pytest tests/ -v
```

---

## 📈 Status Milestone

- [x] Retrain model dengan arsitektur Ultralytics YOLO26 & Export ONNX FP16
- [x] Hyperparameter tuning (AdamW, lr=0.005) & logging ke MLflow
- [x] Perbaikan error DLL & konfigurasi Docker Compose hot-reloading (5 containers)
- [x] Implementasi Prometheus & Grafana Dashboard-as-Code
- [x] Integrasi Grafana & MLflow MCP Server untuk AI Agent
- [x] Implementasi Automated CI/CD Workflow (GitHub Actions ke HF Spaces)
- [x] Paket Showcase & Portofolio Eksekutif (README.md standar dunia, carousel.html 1080x1350, SHOWCASE_PACK.md)
- [x] Sinkronisasi dokumentasi sistem (`SYSTEM_STATE.md` & `walkthrough.md`)
