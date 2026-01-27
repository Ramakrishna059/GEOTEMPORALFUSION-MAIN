import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import random
import pandas as pd
import torchvision.transforms as transforms

# Import your model architecture
from step4_model_architecture import GeoTemporalFusionNet

# --- CONFIGURATION ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = "geo_temporal_model.pth"
CSV_PATH = os.path.join("data", "raw", "fire_locations.csv")
IMG_DIR = os.path.join("data", "raw", "images")
WEATHER_DIR = os.path.join("data", "processed", "weather")
MASK_DIR = os.path.join("data", "processed", "masks")

def visualize_prediction():
    print("--- LOADING MODEL ---")
    # 1. Initialize the model structure
    model = GeoTemporalFusionNet().to(DEVICE)
    
    # 2. Load the trained weights (the "Brain" you saved in Step 5)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval() # Set to evaluation mode (turns off training features)
    print(" [✓] Model loaded successfully.")

    # 3. Pick a random sample from your CSV
    df = pd.read_csv(CSV_PATH)
    random_idx = random.randint(0, len(df) - 1)
    row = df.iloc[random_idx]
    lat, lon = row['latitude'], row['longitude']
    
    print(f"--- PREDICTING FOR SAMPLE {random_idx} ---")
    print(f"Location: {lat}, {lon}")

    # 4. Load and Preprocess Input Data (Same as training)
    # A. Image
    img_name = f"fire_{random_idx}_{lat}_{lon}.jpg"
    img_path = os.path.join(IMG_DIR, img_name)
    original_img = Image.open(img_path).convert("RGB")
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    input_img = transform(original_img).unsqueeze(0).to(DEVICE) # Add batch dimension

    # B. Weather
    weather_name = f"weather_{random_idx}.npy"
    weather_path = os.path.join(WEATHER_DIR, weather_name)
    if os.path.exists(weather_path):
        weather_data = np.load(weather_path).astype(np.float32)
    else:
        weather_data = np.zeros((24, 4), dtype=np.float32)
    input_weather = torch.tensor(weather_data).unsqueeze(0).to(DEVICE)

    # C. Ground Truth Mask (The Answer Key)
    mask_name = f"fire_{random_idx}_{lat}_{lon}_mask.png"
    mask_path = os.path.join(MASK_DIR, mask_name)
    true_mask = Image.open(mask_path).convert("L")
    true_mask = true_mask.resize((256, 256))

    # 5. RUN THE PREDICTION
    with torch.no_grad(): # Don't calculate gradients, just predict
        output_logits = model(input_img, input_weather)
        # Convert logits to probability (0 to 1) using Sigmoid
        prediction_prob = torch.sigmoid(output_logits)
        # Convert to CPU numpy image for plotting
        prediction_map = prediction_prob.squeeze().cpu().numpy()

    # 6. PLOT THE RESULTS
    print("--- GENERATING VISUALIZATION ---")
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Satellite Input
    ax[0].imshow(original_img.resize((256, 256)))
    ax[0].set_title("Satellite Input")
    ax[0].axis('off')
    
    # Plot 2: True Mask (Target)
    ax[1].imshow(true_mask, cmap='gray')
    ax[1].set_title("Ground Truth (Actual Fire)")
    ax[1].axis('off')
    
    # Plot 3: AI Prediction
    # We use a color map 'jet' (Blue=Safe, Red=Fire)
    im = ax[2].imshow(prediction_map, cmap='jet', vmin=0, vmax=1)
    ax[2].set_title("AI Prediction (Risk Map)")
    ax[2].axis('off')
    
    # Add a colorbar to show probability
    fig.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04)
    
    plt.suptitle(f"Geo-Temporal Fusion Result - Sample #{random_idx}", fontsize=16)
    plt.tight_layout()
    plt.show()
    print(" [✓] Visualization displayed.")

if __name__ == "__main__":
    visualize_prediction()