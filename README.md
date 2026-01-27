# GEOTEMPORALFUSION-MAIN

# 🔥 GeoTemporalFusion: AI-Powered Wildfire Prediction System

[![Training Status](https://img.shields.io/badge/Training-100%20Epochs-success)](./models)
[![Accuracy](https://img.shields.io/badge/Accuracy-100%25-brightgreen)](./step5_train.py)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](./LICENSE)

A deep learning system that predicts wildfire risk by fusing satellite imagery and weather data using a GeoTemporal Fusion architecture (CNN + LSTM).

---

## 📊 Training Results

| Metric | Value |
|--------|-------|
| **Final Accuracy** | **100.00%** ✅ |
| Epochs Trained | 100 |
| Dataset Size | 9,999 samples |
| Model Parameters | 12,736,192 |
| Training Time | ~27 minutes |
| Hardware | NVIDIA GTX 1650 (CUDA) |

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the API Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Test the API
```bash
# Health check
curl http://localhost:8000/health

# Model info
curl http://localhost:8000/model/info
```

### 4. Run Tests
```bash
pytest tests/test_suite.py -v
```

---

## 📁 Project Structure

```
GeoTemporalFusion/
├── app/
│   └── main.py               # FastAPI backend
├── models/
│   ├── simple_fire_model.pth # Trained model (100 epochs)
│   └── training_history.json # Training metrics
├── data/
│   ├── raw/                  # Raw satellite images
│   └── processed/            # Processed weather data
├── tests/
│   └── test_suite.py         # Comprehensive test suite
├── step1_get_fires.py        # Fire data collection
├── step2_get_images.py       # Satellite image download
├── step3_process_data.py     # Data preprocessing
├── step4_model_architecture.py # Model definition
├── step5_train.py            # Training script & results
├── step6_visualize.py        # Visualization utilities
├── verify_deploy.py          # Deployment checklist
├── Dockerfile                # Hugging Face deployment
├── requirements.txt          # Python dependencies
└── SRS_DOCUMENT.md           # Full specification
```

---

## 🧠 Model Architecture

### GeoTemporalFusion Network
```
┌─────────────────────────────────────────────────────────────┐
│                    SATELLITE IMAGE (128x128x3)              │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
         ┌────────────────────────────────────┐
         │        IMAGE ENCODER (CNN)         │
         │  Conv2D → MaxPool → Conv2D → Pool  │
         │       128x128 → 8x8 features       │
         └────────────────┬───────────────────┘
                          │
         ┌────────────────┴───────────────────┐
         │                                    │
         ▼                                    ▼
┌─────────────────┐              ┌─────────────────────┐
│  24-HOUR WEATHER │              │   WEATHER ENCODER   │
│  (24 x 4 features)│    ──────▶  │   (MLP: 96 → 64)    │
└─────────────────┘              └──────────┬──────────┘
                                            │
         ┌──────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│                    FUSION LAYER                              │
│              Concatenate(img_features, weather_features)     │
│                      8192 + 64 = 8256 features              │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
         ┌────────────────────────────────────┐
         │           DECODER (MLP)            │
         │     8256 → 512 → 16384 (128x128)   │
         └────────────────┬───────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   FIRE RISK HEATMAP (128x128)               │
│               Values: 0.0 (safe) to 1.0 (high risk)         │
└─────────────────────────────────────────────────────────────┘
```

---

## 🌐 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API documentation (Swagger UI) |
| `/health` | GET | Health check |
| `/model/info` | GET | Model information |
| `/predict` | POST | Fire risk prediction |
| `/predict/image` | POST | Quick prediction from image upload |

### Example Prediction Request
```python
import requests
import base64

# Load and encode image
with open("satellite_image.png", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

# Prepare weather data (24 hours)
weather = {
    "temperature": [30.0] * 24,      # °C
    "humidity": [40.0] * 24,          # %
    "wind_speed": [5.0] * 24,         # m/s
    "wind_direction": [180.0] * 24    # degrees
}

# Make prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={"image": image_b64, "weather": weather}
)
print(response.json())
```

---

## 🚀 Deployment

### Free "Split-Stack" Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                      USER BROWSER                           │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              VERCEL (Free Tier)                             │
│              React/HTML Frontend                            │
│              https://geotemporal.vercel.app                 │
└─────────────────────────┬───────────────────────────────────┘
                          │ API Calls
                          ▼
┌─────────────────────────────────────────────────────────────┐
│           HUGGING FACE SPACES (Free Tier)                   │
│           FastAPI + PyTorch Backend                         │
│           16GB RAM, CPU Inference                           │
│           https://username-geotemporal-api.hf.space         │
└─────────────────────────────────────────────────────────────┘
```

### Deploy to Hugging Face Spaces

1. **Create a Space**: Go to [huggingface.co/spaces](https://huggingface.co/spaces) → "Create new Space" → Select "Docker" SDK

2. **Upload Files**:
   ```bash
   git clone https://huggingface.co/spaces/<username>/geotemporal-api
   cd geotemporal-api
   # Copy project files
   git add .
   git commit -m "Deploy GeoTemporalFusion"
   git push
   ```

3. **Wait for Build**: ~5 minutes

4. **Access API**: `https://<username>-geotemporal-api.hf.space`

### Pre-Flight Checklist
```bash
python verify_deploy.py
```
This script verifies:
- ✅ requirements.txt has all dependencies
- ✅ Model file exists
- ✅ No hardcoded local paths
- ✅ Dockerfile is valid
- ✅ API is importable

---

## 🧪 Testing

### Run Full Test Suite
```bash
# All tests with verbose output
pytest tests/test_suite.py -v

# Specific test class
pytest tests/test_suite.py::TestModelLoading -v

# Stop on first failure
pytest tests/test_suite.py -x
```

### Test Categories
| Category | Tests | Description |
|----------|-------|-------------|
| Model Loading | 7 | Verify .pth file loads correctly |
| Inference Sanity | 6 | Test output shapes and ranges |
| API Endpoints | 4 | Test REST API with TestClient |
| Error Handling | 5 | Verify graceful error handling |
| Integration | 3 | End-to-end pipeline tests |

---

## 📈 Training Details

The model was trained for 100 epochs achieving 100% accuracy:

```
Epoch   Time    Train Loss    Val Loss    Accuracy
─────────────────────────────────────────────────────
1       22.1s   0.021567      0.021021    100.00% ★
25      16.7s   0.020861      0.020879    100.00%
50      15.8s   0.020812      0.020899    100.00%
75      15.8s   0.020765      0.020934    100.00%
100     16.0s   0.020726      0.020977    100.00%
```

See [step5_train.py](./step5_train.py) for complete training history.

---

## 📚 Documentation

- [SRS Document](./SRS_DOCUMENT.md) - Full software requirements specification
- [Model Architecture](./step4_model_architecture.py) - Network definition
- [Training Results](./step5_train.py) - Complete training log

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Run tests: `pytest tests/test_suite.py`
4. Submit a pull request

---

## 📄 License

MIT License - See [LICENSE](./LICENSE) for details.

---

## 👥 Team

**Major Project - Wildfire Prediction System**

Built with ❤️ using PyTorch, FastAPI, and React
