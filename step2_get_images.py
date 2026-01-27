import pandas as pd
import requests
import os
from PIL import Image
from io import BytesIO

# --- CONFIGURATION ---
INPUT_CSV = os.path.join("data", "raw", "fire_locations.csv")
OUTPUT_DIR = os.path.join("data", "raw", "images")

# We will use a public satellite tile server (ArcGIS World Imagery)
# This gives us high-res satellite views for free without a key
TILE_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"

def latlon_to_tile(lat, lon, zoom):
    """
    Converts Lat/Lon to Map Tile coordinates (X, Y) for the zoom level.
    This is standard web mapping math.
    """
    import math
    n = 2.0 ** zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    radians = math.radians(lat)
    ytile = int((1.0 - math.log(math.tan(radians) + (1 / math.cos(radians))) / math.pi) / 2.0 * n)
    return xtile, ytile

def download_images():
    if not os.path.exists(INPUT_CSV):
        print("Error: fire_locations.csv not found. Run Step 1 first.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(INPUT_CSV)
    
    print(f"Found {len(df)} locations. Downloading satellite tiles...")
    
    zoom_level = 15 # High resolution (~4 meters/pixel)
    
    for index, row in df.iterrows():
        lat, lon = row['latitude'], row['longitude']
        
        # Calculate tile coordinates
        x, y = latlon_to_tile(lat, lon, zoom_level)
        
        # Construct URL
        url = TILE_URL.format(z=zoom_level, y=y, x=x)
        
        try:
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            if response.status_code == 200:
                # Save the image
                img = Image.open(BytesIO(response.content))
                filename = f"fire_{index}_{lat}_{lon}.jpg"
                save_path = os.path.join(OUTPUT_DIR, filename)
                img.save(save_path)
                print(f" [✓] Downloaded: {filename}")
            else:
                print(f" [!] Failed {lat}, {lon} (Status: {response.status_code})")
                
        except Exception as e:
            print(f" [!] Error: {e}")

    print("\nDone! Check the 'data/raw/images' folder.")

if __name__ == "__main__":
    download_images()