# SYSTEM STATE - VisionOps Guard

**Last Synchronized:** 2026-09-23 22:20:00 (WIB)  
**System Version:** v1.2.0  
**Environment:** Windows 11 (Host) + Docker Desktop (WSL2 Linux Containers)  
**Deployment Target:** GitHub & Hugging Face Spaces (`https://huggingface.co/spaces/rfahrur6045/mlops-final-copy`)

---

## 📌 Status Ringkasan Proyek

VisionOps Guard adalah sistem MLOps end-to-end berbasis Computer Vision untuk deteksi Alat Pelindung Diri (K3 / PPE) secara real-time. Sistem telah berhasil dikontainerisasi sepenuhnya menggunakan Docker dan siap dioperasikan tanpa hambatan kebijakan pemblokiran DLL pada Windows, lengkap dengan Full Observability Stack (Prometheus + Grafana + MCP Server).

---

## 🛠️ Komponen Sistem & Status Terbaru

| Komponen | Status | Deskripsi & Port |
| :--- | :--- | :--- |
| **ONNX Inference Engine** | 🟢 Active | Quantized FP16 model (`models/visionops_guard.onnx`) dengan latensi ~12.5ms (GPU) / ~23.1ms (CPU). |
| **Streamlit Web UI** | 🟢 Active | UI interaktif di port **8501** (`ui/app.py`), menggunakan `st.table()` untuk performa render cepat. |
| **FastAPI REST Service** | 🟢 Active | Microservice di port **8000** (`src/serving/main:app`) dengan endpoint `/predict`, `/health`, dan `/metrics`. |
| **Docker Compose Stack** | 🟢 Active | Multi-container stack (`visionops-api`, `visionops-ui`, `visionops-prometheus`, `visionops-grafana`) dengan live volume mounts (`.:/app`) & `PYTHONPATH=/app`. |
| **Prometheus Monitoring** | 🟢 Active | Observabilitas matriks time-series di port **9090** (`config/prometheus.yml`) mengambil data dari `visionops-api:8000`. |
| **Grafana Visual Dashboard** | 🟢 Active | Dashboard analitik visual eksekutif di port **3000** (`visionops-grafana`) dengan auto-provisioned Prometheus datasource & pre-built dashboard (*Executive Command Center*). |
| **Grafana MCP Server** | 🟢 Active | Model Context Protocol integration (`@grafana/mcp-server`) di `mcp_config.json` via Service Account Token. |

---

## 🔄 Pembaruan Terakhir (Recent Updates)

1. **Implementasi Full Observability Stack (Grafana + Prometheus):**
   - Ditambahkan container `grafana/grafana:10.4.0` ke `docker-compose.yml` di port `3000`.
   - Dikonfigurasi **Auto-Provisioning Datasource** Prometheus di `config/grafana/provisioning/datasources/prometheus.yml`.
   - Dibuat **Dashboard-as-Code** otomatis di `config/grafana/provisioning/dashboards/visionops_guard.json` (*VisionOps Guard - Executive Command Center*).
   - Diintegrasikan **Grafana MCP Server** (`@grafana/mcp-server`) dengan token otentikasi Service Account ke dalam `mcp_config.json`.
2. **Pengembangan Docker Containerization & Resolusi DLL Windows:**
   - Dibuat `.dockerignore` untuk optimasi konteks build (menolak `.venv`, `.git`, `runs`, `scratch`, `weights`).
   - Diperbarui `Dockerfile` dari paket `libgl1-mesa-glx` ke `libgl1` (kompatibel dengan Debian 12/13).
   - Ditambahkan `PYTHONPATH=/app` dan `PYTHONUNBUFFERED=1` pada `docker-compose.yml` untuk mencegah `ModuleNotFoundError: No module named 'src'`.
   - Mengalihkan eksekusi lokal dari `.venv` ke Docker Container untuk mengeliminasi pemblokiran biner `.pyd` oleh Windows Application Control (AppLocker/WDAC).
   - Mengganti `st.dataframe()` dengan `st.table()` pada `ui/app.py` untuk mengabaikan eksekusi C-extension `pyarrow.lib.pyd`.
3. **Validasi Test Suite & Deployment:**
   - 6/6 unit test (`pytest tests/`) lulus 100%.
   - Push ke Dual-Remote: GitHub (`origin/main`) dan Hugging Face Spaces (`hf main`).

---

## 🚀 Panduan Eksekusi Sistem

### A. Menggunakan Docker Compose (Rekomendasi Utama)
```powershell
docker compose up -d
```
- 🎨 **Streamlit Web UI:** `http://localhost:8501`
- ⚡ **FastAPI REST Docs:** `http://localhost:8000/docs`
- 📊 **Prometheus Engine:** `http://localhost:9090`
- 📈 **Grafana Command Center:** `http://localhost:3000` *(Login: admin / admin)*

### B. Menggunakan Environment Lokal (.venv)
```powershell
.venv\Scripts\python.exe -m streamlit run ui/app.py
```

### C. Menjalankan MLflow UI (Offline Training Tracking)
```powershell
.venv\Scripts\python.exe -m mlflow ui --port 5000
```
- Akses di: `http://localhost:5000`

---

## 📈 Milestone & Task Selanjutnya

- [x] Retrain model YOLO & Export ONNX FP16
- [x] Perbaikan error DLL & konfigurasi Docker Compose hot-reloading
- [x] Implementasi Prometheus & Grafana Dashboard-as-Code
- [x] Integrasi Grafana MCP Server untuk AI Agent
- [x] Sinkronisasi dokumentasi sistem (`SYSTEM_STATE.md` & `README.md`)
- [ ] Implementasi Automated CI/CD Workflow (GitHub Actions ke HF Spaces)
