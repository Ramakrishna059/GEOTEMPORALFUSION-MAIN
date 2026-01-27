import pandas as pd
import requests
import numpy as np
import os
from PIL import Image, ImageDraw

# --- CONFIGURATION ---
INPUT_CSV = os.path.join("data", "raw", "fire_locations.csv")
WEATHER_DIR = os.path.join("data", "processed", "weather")
MASK_DIR = os.path.join("data", "processed", "masks")

# Create directories
os.makedirs(WEATHER_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)

def get_historical_weather(lat, lon, date):
    """
    Fetches hourly weather from Open-Meteo for the specific fire date.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date,
        "end_date": date,
        "hourly": ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_direction_10m"]
    }
    
    try:
        r = requests.get(url, params=params)
        data = r.json()
        
        # Extract the hourly data into a simple array (24 hours x 4 features)
        hourly = data.get("hourly", {})
        if not hourly:
            return None
            
        # Stack them: Temp, Humid, Speed, Direction
        weather_matrix = np.column_stack((
            hourly["temperature_2m"],
            hourly["relative_humidity_2m"],
            hourly["wind_speed_10m"],
            hourly["wind_direction_10m"]
        ))
        # We only want the first 24 hours
        return weather_matrix[:24]
        
    except Exception as e:
        print(f"Weather Error: {e}")
        return None

def create_synthetic_mask(filename):
    """
    Creates a black image with a white blob in the center.
    This represents the 'Fire' for training our AI.
    """
    # Create black image (Background)
    mask = Image.new('L', (256, 256), 0) 
    draw = ImageDraw.Draw(mask)
    
    # Draw a white circle (The Fire) in the middle with random size
    # In a real project, this would be the actual burn scar shape
    import random
    r = random.randint(20, 60) # Radius
    center_x, center_y = 128, 128
    
    # Add some "noise" to the location so the AI learns better
    offset_x = random.randint(-30, 30)
    offset_y = random.randint(-30, 30)
    
    draw.ellipse(
        (center_x + offset_x - r, center_y + offset_y - r, 
         center_x + offset_x + r, center_y + offset_y + r), 
        fill=255
    )
    
    mask.save(os.path.join(MASK_DIR, filename.replace(".jpg", "_mask.png")))

def process_dataset():
    if not os.path.exists(INPUT_CSV):
        print("CSV missing. Run Step 1.")
        return

    df = pd.read_csv(INPUT_CSV)
    print(f"Processing {len(df)} samples...")
    
    success_count = 0

    for index, row in df.iterrows():
        lat, lon = row['latitude'], row['longitude']
        date = row['acq_date']
        
        # 1. Get Weather
        weather = get_historical_weather(lat, lon, date)
        
        if weather is not None:
            # Save Weather as .npy (numpy file)
            weather_filename = f"weather_{index}.npy"
            np.save(os.path.join(WEATHER_DIR, weather_filename), weather)
            
            # 2. Create Mask
            # We assume the image filename matches the index from step 2
            image_filename = f"fire_{index}_{lat}_{lon}.jpg"
            create_synthetic_mask(image_filename)
            
            print(f" [✓] Processed Sample {index}")
            success_count += 1
        else:
            print(f" [!] Failed weather for {index}")

    print("\n------------------------------------------------")
    print(f"DONE! Created {success_count} complete training samples.")
    print("Check 'data/processed/masks' to see the fire targets.")
    print("------------------------------------------------")

if __name__ == "__main__":
    process_dataset()