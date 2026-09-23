# SYSTEM STATE - VisionOps Guard

**Last Synchronized:** 2026-09-23 14:21:00 (WIB)  
**System Version:** v1.1.0  
**Environment:** Windows 11 (Host) + Docker Desktop (WSL2 Linux Containers)  
**Deployment Target:** GitHub & Hugging Face Spaces (`https://huggingface.co/spaces/rfahrur6045/mlops-final-copy`)

---

## 📌 Status Ringkasan Proyek

VisionOps Guard adalah sistem MLOps end-to-end berbasis Computer Vision untuk deteksi Alat Pelindung Diri (K3 / PPE) secara real-time. Sistem telah berhasil dikontainerisasi sepenuhnya menggunakan Docker dan siap dioperasikan tanpa hambatan kebijakan pemblokiran DLL pada Windows.

---

## 🛠️ Komponen Sistem & Status Terbaru

| Komponen | Status | Deskripsi & Port |
| :--- | :--- | :--- |
| **ONNX Inference Engine** | 🟢 Active | Quantized FP16 model (`models/visionops_guard.onnx`) dengan latensi ~12.5ms (GPU) / ~23.1ms (CPU). |
| **Streamlit Web UI** | 🟢 Active | UI interaktif di port **8501** (`ui/app.py`), menggunakan `st.table()` untuk performa render cepat. |
| **FastAPI REST Service** | 🟢 Active | Microservice di port **8000** (`src/serving/main:app`) dengan endpoint `/predict`, `/health`, dan `/metrics`. |
| **Docker Compose Stack** | 🟢 Active | Multi-container stack (`visionops-api`, `visionops-ui`, `visionops-prometheus`) dengan live volume mounts (`.:/app`) & `PYTHONPATH=/app`. |
| **Prometheus Monitoring** | 🟢 Active | Observabilitas matriks di port **9090** (`config/prometheus.yml`). |

---

## 🔄 Pembaruan Terakhir (Recent Updates)

1. **Pengembangan Docker Containerization:**
   - Dibuat `.dockerignore` untuk optimasi konteks build (menolak `.venv`, `.git`, `runs`, `scratch`, `weights`).
   - Diperbarui `Dockerfile` dari paket `libgl1-mesa-glx` ke `libgl1` (kompatibel dengan Debian 12/13).
   - Ditambahkan `PYTHONPATH=/app` dan `PYTHONUNBUFFERED=1` pada `docker-compose.yml` untuk mencegah `ModuleNotFoundError: No module named 'src'`.
2. **Penanganan Pemblokiran DLL Windows:**
   - Mengalihkan eksekusi lokal dari `.venv` ke Docker Container untuk mengeliminasi pemblokiran biner `.pyd` oleh Windows Application Control (AppLocker/WDAC).
   - Mengganti `st.dataframe()` dengan `st.table()` pada `ui/app.py` untuk mengabaikan eksekusi C-extension `pyarrow.lib.pyd`.
3. **Validasi Test Suite:**
   - 6/6 unit test (`pytest tests/`) lulus 100%.

---

## 🚀 Panduan Eksekusi Sistem

### A. Menggunakan Docker (Rekomendasi Utama)
```powershell
docker compose up -d
```
- Streamlit UI: `http://localhost:8501`
- FastAPI REST Docs: `http://localhost:8000/docs`
- Prometheus Metrics: `http://localhost:9090`

### B. Menggunakan Environment Lokal (.venv)
```powershell
.venv\Scripts\python.exe -m streamlit run ui/app.py
```

---

## 📈 Milestone & Task Selanjutnya

- [x] Retrain model YOLO & Export ONNX FP16
- [x] Perbaikan error DLL & konfigurasi Docker Compose hot-reloading
- [x] Sinkronisasi dokumentasi sistem (`SYSTEM_STATE.md`)
- [ ] Implementasi Automated CI/CD Workflow (GitHub Actions ke HF Spaces)
