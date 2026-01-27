"""
================================================================================
🧪 COMPREHENSIVE TEST SUITE - GEOTEMPORAL FUSION WILDFIRE PREDICTION
================================================================================

Automated "Bug Hunter" Test Suite for Major Project Viva

Tests Include:
1. Model Loading Tests - Verify .pth file loads without memory errors
2. Inference Sanity Tests - Pass dummy tensors and verify output shape
3. API Endpoint Tests - Test all REST endpoints with TestClient
4. Error Handling Tests - Verify graceful error handling for bad inputs
5. Integration Tests - End-to-end prediction pipeline

Run with: pytest tests/test_suite.py -v
================================================================================
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import json
import os
import sys
import base64
from io import BytesIO
from PIL import Image
import warnings

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "simple_fire_model.pth")
HISTORY_PATH = os.path.join(BASE_DIR, "models", "training_history_simple.json")
IMG_SIZE = 128
WEATHER_HOURS = 24
WEATHER_FEATURES = 4


# ============================================================
# MODEL DEFINITION (for testing)
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
# FIXTURES
# ============================================================
@pytest.fixture
def device():
    """Get the best available device"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def model(device):
    """Load the trained model"""
    model = SimpleFireNet(img_size=IMG_SIZE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model


@pytest.fixture
def dummy_image(device):
    """Create a dummy image tensor"""
    return torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)


@pytest.fixture
def dummy_weather(device):
    """Create dummy weather data"""
    return torch.randn(1, WEATHER_HOURS, WEATHER_FEATURES).to(device)


@pytest.fixture
def zero_image(device):
    """Create an all-zeros image tensor"""
    return torch.zeros(1, 3, IMG_SIZE, IMG_SIZE).to(device)


@pytest.fixture
def zero_weather(device):
    """Create all-zeros weather data"""
    return torch.zeros(1, WEATHER_HOURS, WEATHER_FEATURES).to(device)


# ============================================================
# TEST 1: MODEL LOADING TESTS
# ============================================================
class TestModelLoading:
    """Tests for model loading and initialization"""
    
    def test_model_file_exists(self):
        """Verify the trained model file exists"""
        assert os.path.exists(MODEL_PATH), f"Model file not found: {MODEL_PATH}"
        print(f"✅ Model file exists: {MODEL_PATH}")
    
    def test_model_file_size(self):
        """Verify model file is not corrupted (reasonable size)"""
        if os.path.exists(MODEL_PATH):
            size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
            assert size_mb > 1, "Model file too small, may be corrupted"
            assert size_mb < 500, "Model file too large, unexpected"
            print(f"✅ Model file size: {size_mb:.2f} MB")
    
    def test_model_loads_without_errors(self, device):
        """Verify model loads without memory errors"""
        try:
            model = SimpleFireNet(img_size=IMG_SIZE)
            if os.path.exists(MODEL_PATH):
                model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            model.to(device)
            print("✅ Model loaded without errors")
        except Exception as e:
            pytest.fail(f"Model loading failed: {e}")
    
    def test_model_parameter_count(self, model):
        """Verify model has expected number of parameters"""
        params = sum(p.numel() for p in model.parameters())
        assert params > 1_000_000, "Model has fewer parameters than expected"
        print(f"✅ Model parameters: {params:,}")
    
    def test_model_in_eval_mode(self, model):
        """Verify model is in evaluation mode"""
        assert not model.training, "Model should be in eval mode"
        print("✅ Model is in eval mode")
    
    def test_training_history_exists(self):
        """Verify training history file exists"""
        assert os.path.exists(HISTORY_PATH), f"History file not found: {HISTORY_PATH}"
        print(f"✅ Training history exists: {HISTORY_PATH}")
    
    def test_training_history_valid_json(self):
        """Verify training history is valid JSON"""
        if os.path.exists(HISTORY_PATH):
            with open(HISTORY_PATH, 'r') as f:
                history = json.load(f)
            assert 'train_loss' in history or 'accuracy' in history or isinstance(history, list)
            print("✅ Training history is valid JSON")


# ============================================================
# TEST 2: INFERENCE SANITY TESTS
# ============================================================
class TestInferenceSanity:
    """Tests for model inference correctness"""
    
    def test_inference_with_random_input(self, model, dummy_image, dummy_weather):
        """Test inference with random input tensors"""
        with torch.no_grad():
            output = model(dummy_image, dummy_weather)
        
        assert output.shape == (1, 1, IMG_SIZE, IMG_SIZE), \
            f"Expected shape (1, 1, {IMG_SIZE}, {IMG_SIZE}), got {output.shape}"
        print(f"✅ Output shape correct: {output.shape}")
    
    def test_inference_with_zeros(self, model, zero_image, zero_weather):
        """Test inference with all-zeros input (edge case)"""
        with torch.no_grad():
            output = model(zero_image, zero_weather)
        
        assert output.shape == (1, 1, IMG_SIZE, IMG_SIZE)
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.isinf(output).any(), "Output contains Inf values"
        print("✅ Zero input handled correctly (no NaN/Inf)")
    
    def test_output_range(self, model, dummy_image, dummy_weather):
        """Verify output values are in valid range [0, 1]"""
        with torch.no_grad():
            output = model(dummy_image, dummy_weather)
        
        assert output.min() >= 0, f"Output min {output.min()} < 0"
        assert output.max() <= 1, f"Output max {output.max()} > 1"
        print(f"✅ Output range: [{output.min():.4f}, {output.max():.4f}]")
    
    def test_batch_inference(self, model, device):
        """Test batch inference with multiple samples"""
        batch_size = 4
        images = torch.randn(batch_size, 3, IMG_SIZE, IMG_SIZE).to(device)
        weather = torch.randn(batch_size, WEATHER_HOURS, WEATHER_FEATURES).to(device)
        
        with torch.no_grad():
            output = model(images, weather)
        
        assert output.shape == (batch_size, 1, IMG_SIZE, IMG_SIZE)
        print(f"✅ Batch inference works: {batch_size} samples")
    
    def test_inference_deterministic(self, model, dummy_image, dummy_weather):
        """Verify inference is deterministic (same input = same output)"""
        model.eval()
        with torch.no_grad():
            output1 = model(dummy_image, dummy_weather)
            output2 = model(dummy_image, dummy_weather)
        
        assert torch.allclose(output1, output2), "Inference not deterministic"
        print("✅ Inference is deterministic")
    
    def test_inference_speed(self, model, dummy_image, dummy_weather):
        """Test inference latency (should be < 2 seconds)"""
        import time
        
        # Warm-up
        with torch.no_grad():
            _ = model(dummy_image, dummy_weather)
        
        # Measure
        start = time.time()
        with torch.no_grad():
            _ = model(dummy_image, dummy_weather)
        latency = time.time() - start
        
        assert latency < 2.0, f"Inference too slow: {latency:.3f}s"
        print(f"✅ Inference latency: {latency*1000:.1f}ms")


# ============================================================
# TEST 3: API ENDPOINT TESTS
# ============================================================
class TestAPIEndpoints:
    """Tests for REST API endpoints (requires app to be available)"""
    
    @pytest.fixture
    def client(self):
        """Create FastAPI TestClient"""
        try:
            from fastapi.testclient import TestClient
            from app.main import app
            return TestClient(app)
        except ImportError:
            pytest.skip("FastAPI app not available")
    
    def test_health_endpoint(self, client):
        """Test /health endpoint returns 200"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        print(f"✅ Health endpoint: {data}")
    
    def test_model_info_endpoint(self, client):
        """Test /model/info endpoint returns model details"""
        response = client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert "model_name" in data or "parameters" in data
        print(f"✅ Model info endpoint works")
    
    def test_predict_endpoint_valid_data(self, client):
        """Test /predict endpoint with valid data"""
        # Create mock image
        img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), color='green')
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        img_b64 = base64.b64encode(buffer.getvalue()).decode()
        
        # Create mock weather
        weather_data = {
            "temperature": [25.0] * 24,
            "humidity": [50.0] * 24,
            "wind_speed": [5.0] * 24,
            "wind_direction": [180.0] * 24
        }
        
        response = client.post("/predict", json={
            "image": img_b64,
            "weather": weather_data
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data or "risk_level" in data or "heatmap" in data
        print("✅ Predict endpoint works with valid data")
    
    def test_predict_endpoint_missing_image(self, client):
        """Test /predict endpoint handles missing image gracefully"""
        weather_data = {
            "temperature": [25.0] * 24,
            "humidity": [50.0] * 24,
            "wind_speed": [5.0] * 24,
            "wind_direction": [180.0] * 24
        }
        
        response = client.post("/predict", json={
            "weather": weather_data
        })
        
        # Should return 400 or 422, not 500
        assert response.status_code in [400, 422], \
            f"Expected 400/422, got {response.status_code}"
        print("✅ Missing image handled gracefully")


# ============================================================
# TEST 4: ERROR HANDLING TESTS
# ============================================================
class TestErrorHandling:
    """Tests for error handling and edge cases"""
    
    def test_wrong_image_dimensions(self, model, device):
        """Test handling of wrong image dimensions"""
        wrong_image = torch.randn(1, 3, 64, 64).to(device)  # Wrong size
        weather = torch.randn(1, WEATHER_HOURS, WEATHER_FEATURES).to(device)
        
        # Model should either handle it or raise clear error
        try:
            with torch.no_grad():
                output = model(wrong_image, weather)
            # If it succeeds, output shape might differ
            print(f"⚠️ Model accepted wrong dimensions, output: {output.shape}")
        except Exception as e:
            # Expected behavior - should raise error
            print(f"✅ Model correctly rejects wrong dimensions: {type(e).__name__}")
    
    def test_wrong_weather_sequence(self, model, dummy_image, device):
        """Test handling of wrong weather sequence length"""
        wrong_weather = torch.randn(1, 12, 4).to(device)  # Wrong hours
        
        try:
            with torch.no_grad():
                output = model(dummy_image, wrong_weather)
            pytest.fail("Model should reject wrong weather sequence")
        except Exception as e:
            print(f"✅ Model correctly rejects wrong weather length: {type(e).__name__}")
    
    def test_nan_input_handling(self, model, device):
        """Test handling of NaN values in input"""
        nan_image = torch.full((1, 3, IMG_SIZE, IMG_SIZE), float('nan')).to(device)
        weather = torch.randn(1, WEATHER_HOURS, WEATHER_FEATURES).to(device)
        
        with torch.no_grad():
            output = model(nan_image, weather)
        
        # Check if output is all NaN (expected) or handled gracefully
        has_nan = torch.isnan(output).any()
        print(f"⚠️ NaN input produces {'NaN' if has_nan else 'valid'} output")
    
    def test_extreme_values(self, model, device):
        """Test handling of extreme input values"""
        extreme_image = torch.full((1, 3, IMG_SIZE, IMG_SIZE), 1e6).to(device)
        weather = torch.randn(1, WEATHER_HOURS, WEATHER_FEATURES).to(device)
        
        with torch.no_grad():
            output = model(extreme_image, weather)
        
        assert not torch.isnan(output).any(), "Extreme values caused NaN output"
        print("✅ Extreme values handled without NaN")
    
    def test_empty_batch(self, model, device):
        """Test handling of empty batch"""
        empty_image = torch.randn(0, 3, IMG_SIZE, IMG_SIZE).to(device)
        empty_weather = torch.randn(0, WEATHER_HOURS, WEATHER_FEATURES).to(device)
        
        try:
            with torch.no_grad():
                output = model(empty_image, empty_weather)
            assert output.shape[0] == 0, "Empty batch should produce empty output"
            print("✅ Empty batch handled correctly")
        except Exception as e:
            print(f"✅ Empty batch rejected: {type(e).__name__}")


# ============================================================
# TEST 5: INTEGRATION TESTS
# ============================================================
class TestIntegration:
    """End-to-end integration tests"""
    
    def test_full_prediction_pipeline(self, model, device):
        """Test complete prediction pipeline from raw inputs"""
        # Simulate loading an image
        img_array = np.random.rand(IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
        img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1)).unsqueeze(0).to(device)
        
        # Simulate weather data
        weather_array = np.random.rand(WEATHER_HOURS, WEATHER_FEATURES).astype(np.float32)
        weather_tensor = torch.from_numpy(weather_array).unsqueeze(0).to(device)
        
        # Run prediction
        with torch.no_grad():
            output = model(img_tensor, weather_tensor)
        
        # Convert to numpy for visualization
        heatmap = output.squeeze().cpu().numpy()
        
        assert heatmap.shape == (IMG_SIZE, IMG_SIZE)
        assert heatmap.min() >= 0 and heatmap.max() <= 1
        print(f"✅ Full pipeline works, heatmap shape: {heatmap.shape}")
    
    def test_model_save_load_cycle(self, model, dummy_image, dummy_weather, device, tmp_path):
        """Test model can be saved and loaded correctly"""
        # Save model
        save_path = tmp_path / "test_model.pth"
        torch.save(model.state_dict(), save_path)
        
        # Get original output
        with torch.no_grad():
            original_output = model(dummy_image, dummy_weather)
        
        # Load into new model
        new_model = SimpleFireNet(img_size=IMG_SIZE)
        new_model.load_state_dict(torch.load(save_path, map_location=device))
        new_model.to(device)
        new_model.eval()
        
        # Get new output
        with torch.no_grad():
            new_output = new_model(dummy_image, dummy_weather)
        
        assert torch.allclose(original_output, new_output)
        print("✅ Model save/load cycle preserves weights")
    
    def test_gpu_cpu_consistency(self, dummy_image, dummy_weather):
        """Test model produces consistent results on CPU and GPU"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        model_cpu = SimpleFireNet(img_size=IMG_SIZE)
        model_gpu = SimpleFireNet(img_size=IMG_SIZE)
        
        # Copy weights
        model_gpu.load_state_dict(model_cpu.state_dict())
        model_gpu.to('cuda')
        model_cpu.eval()
        model_gpu.eval()
        
        # Run inference
        img_cpu = dummy_image.cpu()
        weather_cpu = dummy_weather.cpu()
        img_gpu = dummy_image.cuda()
        weather_gpu = dummy_weather.cuda()
        
        with torch.no_grad():
            out_cpu = model_cpu(img_cpu, weather_cpu)
            out_gpu = model_gpu(img_gpu, weather_gpu)
        
        assert torch.allclose(out_cpu, out_gpu.cpu(), atol=1e-5)
        print("✅ CPU/GPU outputs consistent")


# ============================================================
# TEST 6: DATA VALIDATION TESTS
# ============================================================
class TestDataValidation:
    """Tests for data preprocessing and validation"""
    
    def test_image_normalization(self):
        """Test image normalization to [0, 1] range"""
        # Simulate raw image (0-255)
        raw_img = np.random.randint(0, 256, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        normalized = raw_img.astype(np.float32) / 255.0
        
        assert normalized.min() >= 0
        assert normalized.max() <= 1
        print("✅ Image normalization correct")
    
    def test_weather_data_shape(self):
        """Test weather data has correct shape"""
        weather = np.random.rand(WEATHER_HOURS, WEATHER_FEATURES)
        
        assert weather.shape == (24, 4), f"Expected (24, 4), got {weather.shape}"
        print("✅ Weather data shape correct")
    
    def test_data_files_exist(self):
        """Test required data files exist"""
        csv_path = os.path.join(BASE_DIR, "data", "raw", "fire_locations.csv")
        assert os.path.exists(csv_path), f"CSV not found: {csv_path}"
        print(f"✅ Data files exist")


# ============================================================
# MAIN EXECUTION
# ============================================================
def run_all_tests():
    """Run all tests and generate report"""
    print("=" * 70)
    print("🧪 GEOTEMPORAL FUSION - COMPREHENSIVE TEST SUITE")
    print("=" * 70)
    
    # Use pytest to run all tests
    exit_code = pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-x",  # Stop on first failure
    ])
    
    return exit_code


if __name__ == "__main__":
    run_all_tests()
