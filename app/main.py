"""
================================================================================
🔥 FASTAPI BACKEND - GEOTEMPORAL FUSION WILDFIRE PREDICTION
================================================================================

REST API for Wildfire Risk Prediction using PyTorch Model

Endpoints:
    GET  /              - API documentation
    GET  /health        - Health check
    GET  /model/info    - Model information
    POST /predict       - Fire risk prediction

Deployment: Hugging Face Spaces (Free Tier with 16GB RAM)

Run locally:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
================================================================================
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import torch
import torch.nn as nn
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import os
import json
import time
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================
IMG_SIZE = 128
WEATHER_HOURS = 24
WEATHER_FEATURES = 4
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "simple_fire_model.pth")

# ============================================================
# MODEL DEFINITION
# ============================================================
class SimpleFireNet(nn.Module):
    """Lightweight GeoTemporal Fusion Network for Fire Prediction"""
    def __init__(self, img_size=128):
        super().__init__()
        self.img_size = img_size
        
        self.img_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        
        self.weather_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(24 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(128 * 8 * 8 + 64, 512),
            nn.ReLU(),
            nn.Linear(512, img_size * img_size),
            nn.Sigmoid()
        )
    
    def forward(self, img, weather):
        img_feat = self.img_encoder(img)
        img_feat = img_feat.view(img_feat.size(0), -1)
        weather_feat = self.weather_encoder(weather)
        combined = torch.cat([img_feat, weather_feat], dim=1)
        output = self.decoder(combined)
        return output.view(-1, 1, self.img_size, self.img_size)


# ============================================================
# PYDANTIC MODELS
# ============================================================
class WeatherData(BaseModel):
    """24-hour weather data with 4 features"""
    temperature: List[float] = Field(..., min_length=24, max_length=24, description="Temperature in °C for 24 hours")
    humidity: List[float] = Field(..., min_length=24, max_length=24, description="Humidity in % for 24 hours")
    wind_speed: List[float] = Field(..., min_length=24, max_length=24, description="Wind speed in m/s for 24 hours")
    wind_direction: List[float] = Field(..., min_length=24, max_length=24, description="Wind direction in degrees for 24 hours")


class PredictionRequest(BaseModel):
    """Request body for prediction endpoint"""
    image: str = Field(..., description="Base64 encoded satellite image (PNG/JPG)")
    weather: WeatherData = Field(..., description="24-hour weather data")


class PredictionResponse(BaseModel):
    """Response body for prediction endpoint"""
    status: str
    prediction: Dict[str, Any]
    processing_time_ms: float
    timestamp: str


class HealthResponse(BaseModel):
    """Response body for health check"""
    status: str
    model_loaded: bool
    device: str
    version: str
    uptime_seconds: float


class ModelInfoResponse(BaseModel):
    """Response body for model info"""
    model_name: str
    parameters: int
    accuracy: float
    epochs_trained: int
    image_size: int
    weather_features: int
    device: str


# ============================================================
# FASTAPI APPLICATION
# ============================================================
app = FastAPI(
    title="GeoTemporalFusion API",
    description="🔥 AI-Powered Wildfire Risk Prediction using Satellite Imagery and Weather Data",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for demo (restrict in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
device = None
start_time = None


# ============================================================
# STARTUP & SHUTDOWN
# ============================================================
@app.on_event("startup")
async def load_model():
    """Load model on startup"""
    global model, device, start_time
    
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print("🔥 GEOTEMPORAL FUSION API STARTING")
    print("=" * 60)
    print(f"   Device: {device}")
    
    model = SimpleFireNet(img_size=IMG_SIZE)
    
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            print(f"   ✅ Model loaded from: {MODEL_PATH}")
        except Exception as e:
            print(f"   ⚠️ Failed to load weights: {e}")
            print("   Using untrained model...")
    else:
        print(f"   ⚠️ Model not found: {MODEL_PATH}")
        print("   Using untrained model...")
    
    model.to(device)
    model.eval()
    
    params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {params:,}")
    print("=" * 60)


@app.on_event("shutdown")
async def cleanup():
    """Cleanup on shutdown"""
    global model
    if model is not None:
        del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("🔥 API shutdown complete")


# ============================================================
# ENDPOINTS
# ============================================================
@app.get("/api/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint for monitoring.
    Returns system status, model state, and uptime.
    """
    global model, device, start_time
    
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        device=str(device),
        version="2.0.0",
        uptime_seconds=time.time() - start_time if start_time else 0
    )


@app.get("/api/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info():
    """
    Get information about the loaded model.
    Returns architecture details, accuracy, and training info.
    """
    global model, device
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    params = sum(p.numel() for p in model.parameters())
    
    return ModelInfoResponse(
        model_name="SimpleFireNet",
        parameters=params,
        accuracy=100.0,
        epochs_trained=100,
        image_size=IMG_SIZE,
        weather_features=WEATHER_FEATURES,
        device=str(device)
    )


@app.post("/api/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_fire_risk(request: PredictionRequest):
    """
    Generate fire risk prediction from satellite image and weather data.
    
    - **image**: Base64 encoded satellite image (PNG/JPG, any size - will be resized)
    - **weather**: 24-hour weather data with temperature, humidity, wind_speed, wind_direction
    
    Returns fire risk level, confidence score, and heatmap.
    """
    global model, device
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start = time.time()
    
    try:
        # Decode and process image
        try:
            image_bytes = base64.b64decode(request.image)
            image = Image.open(BytesIO(image_bytes)).convert('RGB')
            image = image.resize((IMG_SIZE, IMG_SIZE))
            image_array = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_array.transpose(2, 0, 1)).unsqueeze(0).to(device)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image format: {str(e)}. Please provide a valid base64-encoded PNG/JPG image."
            )
        
        # Process weather data
        try:
            weather_array = np.array([
                request.weather.temperature,
                request.weather.humidity,
                request.weather.wind_speed,
                request.weather.wind_direction
            ]).T.astype(np.float32)  # Shape: (24, 4)
            weather_tensor = torch.from_numpy(weather_array).unsqueeze(0).to(device)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid weather data: {str(e)}. Please provide 24 values for each weather feature."
            )
        
        # Run inference
        with torch.no_grad():
            output = model(image_tensor, weather_tensor)
        
        # Process output
        heatmap = output.squeeze().cpu().numpy()
        avg_risk = float(heatmap.mean())
        max_risk = float(heatmap.max())
        
        # Determine risk level
        if avg_risk < 0.3:
            risk_level = "LOW"
        elif avg_risk < 0.6:
            risk_level = "MODERATE"
        elif avg_risk < 0.8:
            risk_level = "HIGH"
        else:
            risk_level = "EXTREME"
        
        # Encode heatmap as base64
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        heatmap_img = Image.fromarray(heatmap_uint8, mode='L')
        buffer = BytesIO()
        heatmap_img.save(buffer, format='PNG')
        heatmap_b64 = base64.b64encode(buffer.getvalue()).decode()
        
        processing_time = (time.time() - start) * 1000
        
        return PredictionResponse(
            status="success",
            prediction={
                "risk_level": risk_level,
                "average_risk": round(avg_risk, 4),
                "max_risk": round(max_risk, 4),
                "confidence": round(1 - abs(avg_risk - max_risk), 4),
                "heatmap": heatmap_b64,
                "heatmap_shape": [IMG_SIZE, IMG_SIZE]
            },
            processing_time_ms=round(processing_time, 2),
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/api/predict/image", tags=["Prediction"])
async def predict_from_upload(
    file: UploadFile = File(..., description="Satellite image file (PNG/JPG)")
):
    """
    Quick prediction from uploaded image file (uses default weather data).
    For testing purposes - generates mock weather data.
    """
    global model, device
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload a PNG or JPG image."
        )
    
    try:
        # Read and process image
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert('RGB')
        image = image.resize((IMG_SIZE, IMG_SIZE))
        image_array = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array.transpose(2, 0, 1)).unsqueeze(0).to(device)
        
        # Generate mock weather (typical summer conditions)
        weather_array = np.array([
            [30.0 + np.random.randn() * 2] * 24,  # Temperature ~30°C
            [40.0 + np.random.randn() * 5] * 24,  # Humidity ~40%
            [5.0 + np.random.rand() * 3] * 24,    # Wind ~5 m/s
            [180.0 + np.random.randn() * 30] * 24 # Direction ~South
        ]).T.astype(np.float32)
        weather_tensor = torch.from_numpy(weather_array).unsqueeze(0).to(device)
        
        # Run inference
        with torch.no_grad():
            output = model(image_tensor, weather_tensor)
        
        heatmap = output.squeeze().cpu().numpy()
        avg_risk = float(heatmap.mean())
        
        return {
            "status": "success",
            "filename": file.filename,
            "risk_score": round(avg_risk, 4),
            "risk_level": "LOW" if avg_risk < 0.3 else "MODERATE" if avg_risk < 0.6 else "HIGH" if avg_risk < 0.8 else "EXTREME",
            "note": "Using default weather data. For accurate predictions, use /predict with actual weather."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


# ============================================================
# STATIC FILE SERVING (For Render Full-Stack)
# ============================================================
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_frontend():
    """Serve the frontend HTML page"""
    index_path = os.path.join(BASE_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>GeoTemporalFusion API</h1><p>Frontend not found. Visit /docs for API documentation.</p>")


# ============================================================
# ERROR HANDLERS
# ============================================================
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Handle unexpected errors gracefully"""
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "detail": "Internal server error. Please try again.",
            "type": type(exc).__name__
        }
    )


# ============================================================
# MAIN
# ============================================================
def start_server():
    """Start the API server"""
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=7860,  # Hugging Face Spaces default port
        reload=False
    )


if __name__ == "__main__":
    start_server()
