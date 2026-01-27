import pandas as pd
import requests
import os
import time

# --- CONFIGURATION ---
# NASA FIRMS API URL
BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/country/csv"
# We use VIIRS (Suomi NPP) satellite data (375m resolution)
SOURCE = "VIIRS_SNPP_NRT" 
# Regions to download: USA, Australia, Brazil
COUNTRIES = ["USA", "AUS", "BRA"] 
# Look back 10 days to find recent fires
DAYS = 10 

def get_fire_data(map_key):
    all_fires = []
    
    print(f"Connecting to NASA FIRMS API with key: {map_key[:5]}...")
    
    for country in COUNTRIES:
        # Construct the API URL
        url = f"{BASE_URL}/{map_key}/{SOURCE}/{country}/{DAYS}"
        print(f" -> Fetching active fires for {country}...")
        
        try:
            df = pd.read_csv(url)
            
            # Check if NASA returned an error message inside the CSV
            if 'Error' in df.columns or len(df) < 1:
                print(f"    [!] No data or error for {country}. (Check quota or key)")
                continue

            # Filter: Keep only 'high' confidence fires
            # We want clean data for training, not false alarms
            if 'confidence' in df.columns:
                high_conf_fires = df[df['confidence'] != 'l'] # 'l' is low confidence
                print(f"    [*] Found {len(high_conf_fires)} high-confidence fires.")
                all_fires.append(high_conf_fires)
            else:
                print("    [!] 'confidence' column missing, skipping.")
            
            # Pause for 1 second to be polite to the API
            time.sleep(1)
            
        except Exception as e:
            print(f"    [!] Error downloading {country}: {e}")

    if all_fires:
        combined_df = pd.concat(all_fires)
        # Save the file to the data/raw folder we created
        output_path = os.path.join("data", "raw", "fire_locations.csv")
        combined_df.to_csv(output_path, index=False)
        print("\n------------------------------------------------")
        print(f"SUCCESS! Saved {len(combined_df)} fire events to '{output_path}'")
        print("------------------------------------------------")
    else:
        print("\n[!] Failed to download any fire data. Check your API Key.")

if __name__ == "__main__":
    print("--- STEP 1: DOWNLOAD FIRE COORDINATES ---")
    # 1. Go to https://firms.modaps.eosdis.nasa.gov/api/map_key/
    # 2. Enter email -> Get Key from email
    key = input("Paste your NASA FIRMS Map Key here: ").strip()
    get_fire_data(key)