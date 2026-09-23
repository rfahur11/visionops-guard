"""
VisionOps Guard - FastAPI Production REST Server
Microservice serving real-time safety PPE inspection with Prometheus metrics.
"""

import time
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from src.serving.predictor import SafetyPPEPredictor

app = FastAPI(
    title="VisionOps Guard API",
    description="Production-Ready Computer Vision MLOps Microservice for Safety PPE Inspection",
    version="1.0.0"
)

# Enable CORS for web dashboard access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instrument Prometheus Metrics
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# Lazy-loaded Predictor Singleton
predictor = None


def get_predictor():
    global predictor
    if predictor is None:
        predictor = SafetyPPEPredictor()
    return predictor


@app.get("/", tags=["Health"])
async def root():
    return {
        "service": "VisionOps Guard Microservice",
        "status": "healthy",
        "docs_url": "/docs",
        "metrics_url": "/metrics"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "up",
        "timestamp": time.time()
    }


@app.post("/predict", tags=["Inference"])
async def predict_safety_ppe(file: UploadFile = File(...)):
    """
    Accepts an uploaded image file (JPG/PNG) and returns Safety PPE compliance status,
    detected bounding boxes, and base64 annotated preview image.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a valid image (JPG/PNG).")

    try:
        contents = await file.read()
        engine = get_predictor()
        result = engine.predict(contents)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
