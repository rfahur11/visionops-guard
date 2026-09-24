# 🚀 Social Media Showcase & Distribution Pack — VisionOps Guard

Paket materi promosi dan distribusi media sosial (LinkedIn, Twitter/X, Video Showcase) berstandar eksekutif untuk **VisionOps Guard**.

---

## 📌 1. Draf Postingan LinkedIn Carousel (Formula PAS + Story)

> **Format Unggahan:** Unggah file PDF hasil export dari `docs/carousel_en.html` (Versi Bahasa Inggris) atau `docs/carousel.html` (Versi Bahasa Indonesia) sebagai dokumen Carousel di LinkedIn, lalu sertakan caption di bawah ini:

```markdown
# 🚀 Draf Caption LinkedIn (Bahasa Indonesia)

Berapa biaya yang harus ditanggung industri manufaktur atau konstruksi akibat 1 detik kelalaian Alat Pelindung Diri (APD/K3)?

Di lapangan, audit keselamatan kerja masih sering mengandalkan inspeksi visual manual yang bersifat sporadis. Hasilnya? Pelanggaran seperti pekerja tanpa helm (no_helmet) atau tanpa rompi safety (no_vest) baru disadari setelah terjadi insiden berbahaya atau audit berkala.

Di sisi lain, menerapkan model Computer Vision konvensional (PyTorch unquantized) pada puluhan kamera CCTV edge sering kali terbentur latensi tinggi (>80ms/frame) dan biaya server GPU yang membengkak.

Untuk memecahkan tantangan ini, saya merancang dan membangun:
🛡️ VisionOps Guard — Real-Time Industrial Safety PPE & CVOps Platform.

Sistem ini mentransformasikan pengawasan manual menjadi infrastruktur pengawasan keselamatan otomatis berkecepatan tinggi:
⚡ Upgrade ke Ultralytics YOLO26 (120 layers, 2.37M params) dengan tuning optimizer AdamW.
🚀 Latensi inferensi sub-15ms (1.3 ms/image GPU & 12.5 ms CPU) berkat kuantisasi ONNX Runtime FP16 (4.96 MB).
🧪 Full Observability Stack: Prometheus & Grafana Dashboard-as-Code memantau laju kepatuhan K3 dan metrik P99 secara real-time.
🔄 Automated CI/CD Workflow: Validasi QA dataset, pengujian unit Pytest (6/6 passing 100%), dan auto-deploy ke Hugging Face Spaces via GitHub Actions.

Slide carousel dokumen di atas membedah studi kasus teknis dan arsitektur lengkapnya:
📌 Slide 1-2: Mengapa pendekatan inspeksi manual gagal & komparasi teknis
📌 Slide 3: Blueprint arsitektur sistem 5-tier (Ingestion ➔ MLflow ➔ Serving ➔ Observability ➔ CI/CD)
📌 Slide 4: Antarmuka deteksi real-time & multi-class PPE compliance
📌 Slide 5: Deep dive kode ONNX lazy loading & instrumentasi Prometheus
📌 Slide 6: Hasil audit benchmark (Precision 91.5%, mAP50 72.6%, 1,112 objek terevaluasi)
📌 Slide 7: Live demo link & repositori open source

Bagaimana pendekatan tim rekan-rekan dalam mengatasi tantangan latensi dan pemantauan K3 industri? Mari kita berdiskusi di kolom komentar! 👇

🌐 Live Interactive Demo: https://rfahrur6045-mlops-final-copy.hf.space
💻 GitHub Open Source: https://github.com/rfahur11/visionops-guard

#ComputerVision #MLOps #YOLO26 #DeepLearning #ArtificialIntelligence #SoftwareEngineering #IndustrialSafety #DevOps #OpenSource
```

```markdown
# 🌍 Draf Caption LinkedIn (English — RECOMMENDED for Global Reach & Recruiters)

How much does a 1-second Personal Protective Equipment (PPE) compliance failure cost in high-risk industrial environments?

In the field, safety audits still heavily rely on sporadic manual visual inspections. The result? Critical violations like unhelmeted workers or missing safety vests are only discovered after dangerous incidents or routine spot-checks.

On the technical side, deploying conventional Computer Vision models (unquantized PyTorch) across dozens of edge CCTV feeds hits massive latency bottlenecks (>80ms/frame) and skyrocketing GPU cloud server costs.

To solve this, I designed and built:
🛡️ VisionOps Guard — Real-Time Industrial Safety PPE & CVOps Platform.

This platform transforms manual oversight into an automated, high-velocity safety infrastructure:
⚡ SOTA Upgrade: Powered by Ultralytics YOLO26 (120 layers, 2.37M params) with AdamW optimizer tuning.
🚀 Sub-15ms Latency: Achieved 1.3 ms/image (GPU) & 12.5 ms (CPU) inference via ONNX Runtime FP16 quantization (compressed to 4.96 MB).
🧪 Full Observability Stack: Prometheus & Grafana Dashboard-as-Code tracking real-time PPE compliance rates and P99 latency.
🔄 Automated CI/CD Pipeline: GitHub Actions workflow running data QA, Pytest unit tests (6/6 passing 100%), and auto-deploying via Git LFS to Hugging Face Spaces.

The PDF document carousel attached breaks down the technical case study:
📌 Slide 1-2: Why manual safety audits fail & technical trade-offs
📌 Slide 3: 5-Tier production architecture blueprint (Ingestion ➔ MLflow ➔ Serving ➔ Observability ➔ CI/CD)
📌 Slide 4: Real-time detection UI & multi-class PPE compliance
📌 Slide 5: Code deep-dive on ONNX lazy loading & Prometheus instrumentation
📌 Slide 6: Audited performance benchmarks (91.5% Precision, 72.6% mAP50 across 1,112 evaluated objects)
📌 Slide 7: Live demo link & open-source repository

How does your team handle real-time edge vision latency and industrial safety monitoring? Let’s connect and discuss in the comments! 👇

🌐 Live Interactive Demo: https://rfahrur6045-mlops-final-copy.hf.space
💻 Open-Source GitHub Repo: https://github.com/rfahur11/visionops-guard

#ComputerVision #MLOps #YOLO26 #DeepLearning #ArtificialIntelligence #SoftwareEngineering #IndustrialSafety #DevOps #OpenSource
```

---

## 🐦 2. Draf Twitter / X Thread (7 Tweets)

### Tweet 1 (Hook):
> 🚨 Industrial safety audits are broken. Manual spot-checks leave fatal blindspots, while traditional CV models choke on heavy GPU costs.
> 
> Here’s how I built **VisionOps Guard**: a sub-15ms real-time Safety PPE detection system using Ultralytics YOLO26, ONNX FP16, and Docker MLOps stack 🧵👇

### Tweet 2 (The Model Upgrade):
> 1/7 ⚡ Upgraded the core architecture to the latest Ultralytics YOLO26 Nano (`yolo26n.pt`).
> 
> With AdamW tuning (lr0=0.005) over 15 epochs on an RTX 4060 GPU, the model achieved:
> • 91.5% Precision
> • 67.0% Recall
> • 72.6% mAP50 across 1,112 PPE objects!

### Tweet 3 (Latency & ONNX):
> 2/7 🚀 Why unquantized PyTorch hurts at the edge:
> Native PyTorch takes ~35.4ms on CPU.
> 
> By compiling to ONNX FP16 with `onnxslim` and `opset 18`:
> • Model footprint shrunk by 20% to just 4.96 MB
> • Inference dropped to 1.3 ms (GPU) / 12.5 ms (CPU)
> That's a 2.8x speedup! ⚡

### Tweet 4 (Full Observability Stack):
> 3/7 📊 Model accuracy is only half the battle. What about production health?
> 
> VisionOps Guard includes:
> • Prometheus scraping FastAPI `/metrics`
> • Grafana Dashboard-as-Code (:3000) for real-time P99 latency & compliance rates
> • MLflow (:5000) for experiment lineage & artifact storage.

### Tweet 5 (CI/CD to Hugging Face):
> 4/7 🔄 Seamless CI/CD:
> Every `git push origin main` triggers a 2-stage GitHub Actions workflow:
> 1️⃣ Data QA integrity audit & 6/6 Pytest tests (100% passed)
> 2️⃣ Git LFS push mirror straight to Hugging Face Spaces ZeroGPU!

### Tweet 6 (Live Demo):
> 5/7 🎨 Try it live without installing anything:
> Upload any construction or factory image and watch the multi-class PPE audit run in real time:
> 🔗 https://rfahrur6045-mlops-final-copy.hf.space

### Tweet 7 (Wrap & Repo):
> 6/7 💻 The entire project is open-source (MIT License) with full Docker Compose orchestration:
> 
> ⭐ Star the repo: https://github.com/rfahur11/visionops-guard
> 👤 Built by Fahrur Rozi K (@rfahur11)
> 
> RT if you love end-to-end MLOps! 🔁

---

## 🎬 3. Video Demo Script (20-30 Detik Screen Recording)

| Timestamp | Visual Screen Action | Audio / Voiceover / Text Overlay |
| :---: | :--- | :--- |
| **00:00 - 00:04** | Buka layar kosong Streamlit UI / Gradio Hugging Face Spaces. Tampilkan teks hook besar. | **Text Overlay:** *Manual K3 safety audit is slow. What if AI audits PPE in 1.3ms?* |
| **00:04 - 00:09** | Seret (*drag & drop*) foto pekerja konstruksi ke dalam kotak upload gambar. | **Text Overlay:** *Inputting industrial worker image...* |
| **00:09 - 00:15** | Klik tombol **"Inspect Safety Compliance"**. Muncul bounding box berwarna: Helm (95%), Rompi (89%), Sepatu Safety (97%). Badge hijau: **"COMPLIANT ✅"**. | **Text Overlay:** *Instant Multi-PPE Detection: Helmet, Vest, Boots verified!* |
| **00:15 - 00:22** | Beralih tab ke **Grafana Command Center** (`http://localhost:3000`). Tunjukkan grafik Prometheus live metric request throughput melonjak dan latency P99 tercatat 12.5ms. | **Text Overlay:** *Full Observability: Real-Time Prometheus & Grafana Metrics.* |
| **00:22 - 00:28** | Tampilkan tampilan repositori GitHub dengan badge CI/CD passing dan link Hugging Face Spaces. | **Text Overlay:** *Open-source on GitHub. Live on Hugging Face Spaces!* |

---

## 🖨️ 4. Panduan Ekspor Carousel ke PDF LinkedIn (1-Klik)

1. Buka file `docs/carousel.html` di browser Google Chrome atau Microsoft Edge:
   - Path absolut: `file:///d:/porto/visionops-guard/docs/carousel.html`
2. Klik tombol **"Save as PDF (LinkedIn Carousel)"** di control bar atas (atau tekan `Ctrl + P`).
3. Pada menu print browser:
   - **Destination:** *Save as PDF*
   - **Layout:** *Portrait*
   - **Paper Size:** *Letter / A4* (CSS otomatis mengunci ukuran ke 1080x1350px)
   - **Margins:** *None*
   - **Scale:** *100%* atau *Default*
   - **Options:** Centang **"Background graphics"**
4. Klik **Save** dengan nama `VisionOps_Guard_LinkedIn_Carousel.pdf`.
5. Buka LinkedIn ➔ Buat Postingan ➔ Klik ikon Dokumen (📄) ➔ Unggah file PDF tersebut.
