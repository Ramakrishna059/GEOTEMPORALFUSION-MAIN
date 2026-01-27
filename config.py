"""
Centralized Configuration for Geo-Temporal Fusion Project
All paths and hyperparameters defined here
"""

import os
import torch

# ============== DEVICE CONFIGURATION ==============
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# ============== PATH CONFIGURATION ==============
# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# Data files
CSV_PATH = os.path.join(RAW_DATA_DIR, "fire_locations.csv")
IMG_DIR = os.path.join(RAW_DATA_DIR, "images")
WEATHER_DIR = os.path.join(PROCESSED_DATA_DIR, "weather")
MASK_DIR = os.path.join(PROCESSED_DATA_DIR, "masks")

# Model
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "geo_temporal_model.pth")

# ============== TRAINING HYPERPARAMETERS ==============
# For 10,000+ samples (mega dataset) on GPU
BATCH_SIZE = 32             # Increased batch size for GPU (4x faster with GTX 1650)
LEARNING_RATE = 0.0005      # Reduced LR for stable training with more data
EPOCHS = 100                # Extended training for better convergence
NUM_WORKERS = 4             # Data loader workers (GPU can handle parallel loading)
VALIDATION_SPLIT = 0.2      # Use 20% for validation, 80% for training
EARLY_STOPPING_PATIENCE = 10 # Stop if validation loss doesn't improve for 10 epochs

# ============== IMAGE PROCESSING ==============
IMAGE_SIZE = (256, 256)     # Target image size
IMAGE_CHANNELS = 3          # RGB channels

# ============== WEATHER DATA ==============
WEATHER_TIME_STEPS = 24     # 24 hours of weather data
WEATHER_FEATURES = 4        # Temp, Humidity, Wind Speed, Wind Direction

# ============== DATA SCALING (10,000+ SAMPLES) ==============
TOTAL_SAMPLES = 10000       # Mega dataset size
TRAINING_SAMPLES = int(TOTAL_SAMPLES * (1 - VALIDATION_SPLIT))  # 8000 samples
VALIDATION_SAMPLES = int(TOTAL_SAMPLES * VALIDATION_SPLIT)      # 2000 samples
ESTIMATED_TRAINING_TIME = "2-4 hours"  # Depends on GPU/CPU
DATASET_REGIONS = ["California", "Australia", "Brazil", "Indonesia", "Central Africa", "Mediterranean"]

# ============== MODEL ARCHITECTURE ==============
MODEL_NAME = "GeoTemporalFusionNet"
WEATHER_HIDDEN_SIZE = 128   # LSTM hidden size
ENCODER_CHANNELS = [64, 128, 256, 512]  # Encoder channel progression

# ============== API CONFIGURATION ==============
API_HOST = "0.0.0.0"
API_PORT = 8000
API_DEBUG = False

# ============== NASA FIRMS API ==============
NASA_FIRMS_BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/country/csv"
NASA_FIRMS_SOURCE = "VIIRS_SNPP_NRT"
NASA_FIRMS_COUNTRIES = ["USA", "AUS", "BRA"]
NASA_FIRMS_DAYS = 10

# ============== SATELLITE TILE SERVER ==============
TILE_SERVER_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
TILE_ZOOM_LEVEL = 15       # High resolution (~4 meters/pixel)

# ============== CREATE ALL REQUIRED DIRECTORIES ==============
def create_directories():
    """Create all required directories if they don't exist"""
    directories = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        IMG_DIR,
        WEATHER_DIR,
        MASK_DIR
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Directory ready: {directory}")

# Auto-create directories on import
if __name__ != "__main__":
    create_directories()
