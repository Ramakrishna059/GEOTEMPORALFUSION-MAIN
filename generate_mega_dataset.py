"""
MEGA Dataset Generator v3.0 - NASA FIRMS Realistic Patterns
============================================================
Generates 10,000 synthetic datasets based on real-world wildfire patterns:
- Fire locations based on NASA FIRMS data structure
- Satellite imagery with realistic vegetation, terrain, fire signatures
- Weather time-series with diurnal patterns and fire-conducive conditions
- Binary fire masks for ground truth
"""

import os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import json
from tqdm import tqdm
from datetime import datetime, timedelta
import random
import math

# Import configuration
from config import (
    CSV_PATH, IMG_DIR, WEATHER_DIR, MASK_DIR,
    IMAGE_SIZE, WEATHER_TIME_STEPS, TOTAL_SAMPLES
)


class NASAFIRMSDataGenerator:
    """
    Production-grade dataset generator mimicking NASA FIRMS data structure.
    Generates realistic fire detection data with satellite-like patterns.
    """
    
    def __init__(self):
        self.num_samples = TOTAL_SAMPLES
        self.img_size = IMAGE_SIZE
        self.weather_steps = WEATHER_TIME_STEPS
        
        # Ensure directories exist
        os.makedirs(IMG_DIR, exist_ok=True)
        os.makedirs(WEATHER_DIR, exist_ok=True)
        os.makedirs(MASK_DIR, exist_ok=True)
        
        # Real-world fire-prone regions with realistic metadata
        # Based on NASA FIRMS global fire hotspot data
        self.fire_regions = [
            {
                "name": "California_USA",
                "lat_range": (32.5, 42.0),
                "lng_range": (-124.5, -114.0),
                "vegetation": ["Chaparral", "Coniferous Forest", "Grassland", "Mixed Forest"],
                "fire_season": [6, 7, 8, 9, 10, 11],
                "avg_temp": 28,
                "avg_humidity": 25,
                "fire_probability": 0.45
            },
            {
                "name": "Australia_Queensland",
                "lat_range": (-28.0, -10.0),
                "lng_range": (138.0, 154.0),
                "vegetation": ["Eucalyptus Forest", "Savanna", "Tropical Forest", "Grassland"],
                "fire_season": [9, 10, 11, 12, 1, 2],
                "avg_temp": 35,
                "avg_humidity": 20,
                "fire_probability": 0.50
            },
            {
                "name": "Amazon_Brazil",
                "lat_range": (-15.0, 5.0),
                "lng_range": (-75.0, -45.0),
                "vegetation": ["Tropical Rainforest", "Savanna", "Wetland"],
                "fire_season": [7, 8, 9, 10, 11],
                "avg_temp": 32,
                "avg_humidity": 60,
                "fire_probability": 0.40
            },
            {
                "name": "Mediterranean_Europe",
                "lat_range": (36.0, 46.0),
                "lng_range": (-10.0, 35.0),
                "vegetation": ["Mediterranean Shrubland", "Pine Forest", "Oak Forest", "Grassland"],
                "fire_season": [6, 7, 8, 9],
                "avg_temp": 30,
                "avg_humidity": 30,
                "fire_probability": 0.35
            },
            {
                "name": "Siberia_Russia",
                "lat_range": (50.0, 70.0),
                "lng_range": (60.0, 170.0),
                "vegetation": ["Boreal Forest", "Taiga", "Tundra"],
                "fire_season": [5, 6, 7, 8],
                "avg_temp": 22,
                "avg_humidity": 45,
                "fire_probability": 0.38
            },
            {
                "name": "Indonesia_Kalimantan",
                "lat_range": (-5.0, 7.0),
                "lng_range": (95.0, 141.0),
                "vegetation": ["Tropical Peatland", "Rainforest", "Palm Plantation"],
                "fire_season": [7, 8, 9, 10, 11],
                "avg_temp": 30,
                "avg_humidity": 75,
                "fire_probability": 0.42
            },
            {
                "name": "Canada_BritishColumbia",
                "lat_range": (48.0, 60.0),
                "lng_range": (-140.0, -110.0),
                "vegetation": ["Boreal Forest", "Temperate Rainforest", "Subalpine Forest"],
                "fire_season": [5, 6, 7, 8, 9],
                "avg_temp": 25,
                "avg_humidity": 35,
                "fire_probability": 0.40
            },
            {
                "name": "Africa_Congo",
                "lat_range": (-15.0, 10.0),
                "lng_range": (10.0, 35.0),
                "vegetation": ["Tropical Savanna", "Rainforest", "Grassland", "Woodland"],
                "fire_season": [6, 7, 8, 9, 10],
                "avg_temp": 28,
                "avg_humidity": 55,
                "fire_probability": 0.48
            },
            {
                "name": "India_CentralForests",
                "lat_range": (18.0, 28.0),
                "lng_range": (75.0, 88.0),
                "vegetation": ["Deciduous Forest", "Sal Forest", "Teak Forest", "Scrubland"],
                "fire_season": [2, 3, 4, 5, 6],
                "avg_temp": 38,
                "avg_humidity": 25,
                "fire_probability": 0.35
            },
            {
                "name": "Chile_Patagonia",
                "lat_range": (-55.0, -35.0),
                "lng_range": (-76.0, -66.0),
                "vegetation": ["Temperate Rainforest", "Grassland", "Shrubland"],
                "fire_season": [12, 1, 2, 3],
                "avg_temp": 24,
                "avg_humidity": 40,
                "fire_probability": 0.32
            }
        ]
        
        # NASA FIRMS-like satellite sources
        self.satellite_sources = [
            {"name": "VIIRS_SNPP", "resolution": 375, "confidence_base": 0.85},
            {"name": "VIIRS_NOAA20", "resolution": 375, "confidence_base": 0.87},
            {"name": "VIIRS_NOAA21", "resolution": 375, "confidence_base": 0.88},
            {"name": "MODIS_Aqua", "resolution": 1000, "confidence_base": 0.82},
            {"name": "MODIS_Terra", "resolution": 1000, "confidence_base": 0.80},
            {"name": "Sentinel-2_MSI", "resolution": 20, "confidence_base": 0.90},
            {"name": "Landsat-8_OLI", "resolution": 30, "confidence_base": 0.88},
            {"name": "Landsat-9_OLI2", "resolution": 30, "confidence_base": 0.89}
        ]
        
    def generate_realistic_satellite_image(self, fire_intensity=0.0, region=None, time_of_day="day"):
        """
        Generate photorealistic satellite imagery with terrain, vegetation, fire signatures
        """
        width, height = self.img_size
        
        # Generate base terrain
        terrain_noise = np.random.randint(40, 100, (height, width, 3), dtype=np.uint8)
        
        # Vegetation layer based on region
        vegetation_color = [34, 139, 34]
        if region:
            veg_str = str(region.get("vegetation", []))
            if "Savanna" in veg_str or "Grassland" in veg_str:
                vegetation_color = [154, 205, 50]
            elif "Tropical" in veg_str:
                vegetation_color = [0, 100, 0]
            elif "Boreal" in veg_str or "Taiga" in veg_str:
                vegetation_color = [46, 139, 87]
        
        # Apply vegetation patterns
        for c in range(3):
            terrain_noise[:, :, c] = np.clip(
                terrain_noise[:, :, c] * 0.3 + vegetation_color[c] * 0.7 + 
                np.random.normal(0, 15, (height, width)),
                0, 255
            ).astype(np.uint8)
        
        # Add water bodies (5% chance)
        if random.random() < 0.05:
            x = random.randint(0, width)
            for y in range(height):
                x += random.randint(-3, 3)
                x = max(0, min(width-1, x))
                for dx in range(-5, 6):
                    if 0 <= x + dx < width:
                        terrain_noise[y, x + dx] = [65, 105, 225]
        
        # Add roads (10% chance)
        if random.random() < 0.10:
            for _ in range(random.randint(1, 3)):
                if random.random() < 0.5:
                    y = random.randint(0, height-1)
                    terrain_noise[max(0, y-1):min(height, y+2), :] = [128, 128, 128]
                else:
                    x = random.randint(0, width-1)
                    terrain_noise[:, max(0, x-1):min(width, x+2)] = [128, 128, 128]
        
        img = Image.fromarray(terrain_noise, 'RGB')
        
        # Day/night adjustment
        if time_of_day == "night":
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(0.3)
        
        # Cloud layer (30% chance)
        cloud_probability = 0.30
        if region and "Tropical" in str(region.get("vegetation", [])):
            cloud_probability = 0.50
            
        if random.random() < cloud_probability:
            cloud_layer = Image.new('RGBA', self.img_size, (255, 255, 255, 0))
            cloud_draw = ImageDraw.Draw(cloud_layer)
            for _ in range(random.randint(2, 8)):
                x, y = random.randint(0, width), random.randint(0, height)
                radius = random.randint(30, 120)
                opacity = random.randint(80, 200)
                cloud_draw.ellipse([x-radius, y-radius, x+radius, y+radius], 
                                  fill=(255, 255, 255, opacity))
            cloud_layer = cloud_layer.filter(ImageFilter.GaussianBlur(radius=20))
            img = Image.alpha_composite(img.convert('RGBA'), cloud_layer).convert('RGB')
        
        # FIRE SIGNATURE
        if fire_intensity > 0.1:
            fire_layer = Image.new('RGBA', self.img_size, (0, 0, 0, 0))
            fire_draw = ImageDraw.Draw(fire_layer)
            
            num_hotspots = max(1, int(fire_intensity * 20))
            
            for _ in range(num_hotspots):
                x = random.randint(30, width - 30)
                y = random.randint(30, height - 30)
                
                core_radius = int(random.uniform(5, 25) * fire_intensity)
                outer_radius = int(core_radius * 2.5)
                
                if fire_intensity > 0.7:
                    fire_draw.ellipse([x-core_radius//3, y-core_radius//3, 
                                      x+core_radius//3, y+core_radius//3],
                                     fill=(255, 255, 200, 255))
                
                fire_draw.ellipse([x-core_radius//2, y-core_radius//2, 
                                  x+core_radius//2, y+core_radius//2],
                                 fill=(255, 200, 0, int(230 * fire_intensity)))
                
                fire_draw.ellipse([x-core_radius, y-core_radius, 
                                  x+core_radius, y+core_radius],
                                 fill=(255, 100, 0, int(200 * fire_intensity)))
                
                fire_draw.ellipse([x-outer_radius, y-outer_radius, 
                                  x+outer_radius, y+outer_radius],
                                 fill=(200, 30, 0, int(150 * fire_intensity)))
            
            fire_layer = fire_layer.filter(ImageFilter.GaussianBlur(radius=5))
            img = Image.alpha_composite(img.convert('RGBA'), fire_layer).convert('RGB')
            
            # Burned area
            if fire_intensity > 0.3:
                burn_layer = Image.new('RGBA', self.img_size, (0, 0, 0, 0))
                burn_draw = ImageDraw.Draw(burn_layer)
                for _ in range(int(fire_intensity * 15)):
                    x = random.randint(0, width)
                    y = random.randint(0, height)
                    radius = random.randint(10, 50)
                    burn_draw.ellipse([x-radius, y-radius, x+radius, y+radius],
                                     fill=(30, 30, 30, int(100 * fire_intensity)))
                burn_layer = burn_layer.filter(ImageFilter.GaussianBlur(radius=10))
                img = Image.alpha_composite(img.convert('RGBA'), burn_layer).convert('RGB')
            
            # Smoke plumes
            if fire_intensity > 0.4:
                smoke_layer = Image.new('RGBA', self.img_size, (0, 0, 0, 0))
                smoke_draw = ImageDraw.Draw(smoke_layer)
                wind_angle = random.uniform(0, 2 * math.pi)
                
                for _ in range(random.randint(3, 8)):
                    x = random.randint(30, width - 30)
                    y = random.randint(30, height - 30)
                    
                    for drift in range(random.randint(3, 8)):
                        drift_x = x + int(drift * 15 * math.cos(wind_angle))
                        drift_y = y + int(drift * 15 * math.sin(wind_angle))
                        radius = random.randint(30, 80)
                        opacity = max(20, int(120 * fire_intensity * (1 - drift * 0.1)))
                        smoke_draw.ellipse([drift_x-radius, drift_y-radius, 
                                           drift_x+radius, drift_y+radius],
                                          fill=(150, 150, 150, opacity))
                
                smoke_layer = smoke_layer.filter(ImageFilter.GaussianBlur(radius=30))
                img = Image.alpha_composite(img.convert('RGBA'), smoke_layer).convert('RGB')
        
        return img
    
    def generate_fire_mask(self, fire_intensity=0.0):
        """Generate binary fire segmentation mask"""
        width, height = self.img_size
        mask = np.zeros((height, width), dtype=np.uint8)
        
        if fire_intensity > 0.1:
            num_fire_clusters = max(1, int(fire_intensity * 12))
            
            for _ in range(num_fire_clusters):
                center_x = random.randint(30, width - 30)
                center_y = random.randint(30, height - 30)
                
                num_sub_fires = random.randint(3, 10)
                for _ in range(num_sub_fires):
                    sub_x = center_x + random.randint(-30, 30)
                    sub_y = center_y + random.randint(-30, 30)
                    
                    sub_x = max(10, min(width - 10, sub_x))
                    sub_y = max(10, min(height - 10, sub_y))
                    
                    # Ensure valid range for blob dimensions (minimum 5)
                    max_blob_size = max(6, int(30 * fire_intensity))
                    blob_w = random.randint(5, max_blob_size)
                    blob_h = random.randint(5, max_blob_size)
                    
                    for dy in range(-blob_h, blob_h + 1):
                        for dx in range(-blob_w, blob_w + 1):
                            if (dx**2 / max(1, blob_w**2) + dy**2 / max(1, blob_h**2)) <= 1:
                                ny, nx = sub_y + dy, sub_x + dx
                                if 0 <= ny < height and 0 <= nx < width:
                                    mask[ny, nx] = 255
        
        mask_img = Image.fromarray(mask, mode='L')
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=1))
        mask_array = np.array(mask_img)
        mask_array = (mask_array > 127).astype(np.uint8) * 255
        
        return Image.fromarray(mask_array, mode='L')
    
    def generate_weather_timeseries(self, fire_intensity=0.0, region=None):
        """Generate 24-hour weather time series"""
        hours = self.weather_steps
        
        if region:
            base_temp = region.get("avg_temp", 25)
            base_humidity = region.get("avg_humidity", 50)
        else:
            base_temp = 25
            base_humidity = 50
        
        hour_array = np.arange(hours)
        
        temp_amplitude = random.uniform(8, 15)
        temp_phase = 14
        temperature = base_temp + temp_amplitude * np.sin(
            2 * np.pi * (hour_array - temp_phase + 6) / 24
        )
        temperature += np.random.normal(0, 1.5, hours)
        
        humidity = base_humidity - 0.8 * (temperature - base_temp)
        humidity += np.random.normal(0, 5, hours)
        humidity = np.clip(humidity, 5, 98)
        
        if fire_intensity > 0.3:
            temperature += fire_intensity * 8
            humidity -= fire_intensity * 20
            humidity = np.clip(humidity, 5, 50)
        
        base_wind = random.uniform(5, 15)
        wind_amplitude = random.uniform(5, 15)
        wind_speed = base_wind + wind_amplitude * np.sin(
            2 * np.pi * (hour_array - 15) / 24
        )
        wind_speed = np.abs(wind_speed) + np.random.normal(0, 3, hours)
        
        if fire_intensity > 0.5:
            wind_speed += fire_intensity * 20
            gust_indices = random.sample(range(hours), k=min(5, random.randint(2, 5)))
            wind_speed[gust_indices] += random.uniform(15, 40)
        
        wind_speed = np.clip(wind_speed, 0, 100)
        
        wind_direction = np.zeros(hours)
        current_dir = random.uniform(0, 360)
        for h in range(hours):
            current_dir += random.uniform(-20, 20)
            wind_direction[h] = current_dir % 360
        
        weather_data = np.column_stack([
            temperature,
            humidity,
            wind_speed,
            wind_direction
        ])
        
        return weather_data.astype(np.float32)
    
    def generate_fire_location_data(self):
        """Generate fire location CSV matching NASA FIRMS data structure"""
        fire_data = []
        current_month = datetime.now().month
        
        print("📍 Generating fire location metadata...")
        
        for i in tqdm(range(self.num_samples), desc="Fire Locations"):
            region_weights = [r["fire_probability"] for r in self.fire_regions]
            total_weight = sum(region_weights)
            region_weights = [w / total_weight for w in region_weights]
            region = random.choices(self.fire_regions, weights=region_weights, k=1)[0]
            
            latitude = random.uniform(*region["lat_range"])
            longitude = random.uniform(*region["lng_range"])
            
            in_fire_season = current_month in region.get("fire_season", range(1, 13))
            base_fire_prob = region["fire_probability"]
            fire_prob = base_fire_prob * 1.5 if in_fire_season else base_fire_prob * 0.5
            
            has_fire = random.random() < fire_prob
            
            if has_fire:
                fire_intensity = np.random.beta(2, 3)
                fire_intensity = 0.2 + fire_intensity * 0.8
            else:
                fire_intensity = random.uniform(0.0, 0.15)
            
            satellite = random.choice(self.satellite_sources)
            
            confidence = satellite["confidence_base"] + random.uniform(-0.1, 0.1)
            if fire_intensity > 0.7:
                confidence += 0.05
            confidence = min(0.99, max(0.50, confidence))
            
            days_ago = random.randint(0, 365)
            fire_date = datetime.now() - timedelta(days=days_ago)
            
            brightness_temp = 300 + fire_intensity * 900
            frp = fire_intensity * random.uniform(50, 500)
            
            hour = random.randint(0, 23)
            daynight = "D" if 6 <= hour <= 18 else "N"
            
            vegetation = random.choice(region["vegetation"])
            
            fire_data.append({
                "fire_id": f"FIRE_{i:05d}",
                "latitude": round(latitude, 6),
                "longitude": round(longitude, 6),
                "brightness": round(brightness_temp, 1),
                "scan": round(random.uniform(0.4, 1.2), 2),
                "track": round(random.uniform(0.4, 1.2), 2),
                "acq_date": fire_date.strftime("%Y-%m-%d"),
                "acq_time": f"{hour:02d}{random.randint(0, 59):02d}",
                "satellite": satellite["name"],
                "instrument": satellite["name"].split("_")[0],
                "confidence": round(confidence * 100),
                "version": "2.0NRT",
                "bright_t31": round(brightness_temp - random.uniform(20, 50), 1),
                "frp": round(frp, 1),
                "daynight": daynight,
                "type": 0 if fire_intensity < 0.2 else random.choice([0, 1, 2, 3]),
                "fire_intensity": round(fire_intensity, 4),
                "region": region["name"],
                "vegetation": vegetation,
                "time_of_day": "day" if daynight == "D" else "night"
            })
        
        return pd.DataFrame(fire_data)
    
    def generate_complete_dataset(self):
        """Main generation pipeline"""
        print("=" * 80)
        print("🔥 NASA FIRMS-STYLE MEGA DATASET GENERATOR v3.0")
        print("=" * 80)
        print(f"   Target Samples: {self.num_samples:,}")
        print(f"   Image Size: {self.img_size[0]}x{self.img_size[1]} pixels")
        print(f"   Weather Time Steps: {self.weather_steps} hours")
        print(f"   Regions: {len(self.fire_regions)} global fire-prone zones")
        print(f"   Satellite Sources: {len(self.satellite_sources)} sensors")
        print("=" * 80)
        print()
        
        # Step 1: Generate fire coordinates CSV
        print("📍 STEP 1/4: Generating fire location metadata...")
        fire_df = self.generate_fire_location_data()
        fire_df.to_csv(CSV_PATH, index=False)
        print(f"   ✅ Saved {len(fire_df):,} fire records to {CSV_PATH}")
        
        fire_samples = len(fire_df[fire_df['fire_intensity'] > 0.2])
        no_fire_samples = len(fire_df[fire_df['fire_intensity'] <= 0.2])
        print(f"   📊 Fire samples: {fire_samples:,} | No-fire samples: {no_fire_samples:,}")
        print()
        
        # Step 2: Generate satellite images
        print("🛰️  STEP 2/4: Generating satellite imagery...")
        for idx, row in tqdm(fire_df.iterrows(), total=len(fire_df), desc="Satellite Images"):
            region = next((r for r in self.fire_regions if r['name'] == row['region']), None)
            img = self.generate_realistic_satellite_image(
                fire_intensity=row['fire_intensity'],
                region=region,
                time_of_day=row['time_of_day']
            )
            img_path = os.path.join(IMG_DIR, f"{row['fire_id']}.png")
            img.save(img_path, quality=95)
        print(f"   ✅ Saved {self.num_samples:,} satellite images to {IMG_DIR}")
        print()
        
        # Step 3: Generate fire masks
        print("🔥 STEP 3/4: Generating fire segmentation masks...")
        for idx, row in tqdm(fire_df.iterrows(), total=len(fire_df), desc="Fire Masks"):
            mask = self.generate_fire_mask(fire_intensity=row['fire_intensity'])
            mask_path = os.path.join(MASK_DIR, f"{row['fire_id']}_mask.png")
            mask.save(mask_path)
        print(f"   ✅ Saved {self.num_samples:,} fire masks to {MASK_DIR}")
        print()
        
        # Step 4: Generate weather time series
        print("🌡️  STEP 4/4: Generating weather time-series data...")
        for idx, row in tqdm(fire_df.iterrows(), total=len(fire_df), desc="Weather Data"):
            region = next((r for r in self.fire_regions if r['name'] == row['region']), None)
            weather_data = self.generate_weather_timeseries(
                fire_intensity=row['fire_intensity'],
                region=region
            )
            weather_path = os.path.join(WEATHER_DIR, f"{row['fire_id']}_weather.npy")
            np.save(weather_path, weather_data)
        print(f"   ✅ Saved {self.num_samples:,} weather files to {WEATHER_DIR}")
        print()
        
        # Calculate disk usage
        import glob
        img_files = glob.glob(os.path.join(IMG_DIR, "*.png"))
        mask_files = glob.glob(os.path.join(MASK_DIR, "*.png"))
        weather_files = glob.glob(os.path.join(WEATHER_DIR, "*.npy"))
        
        total_size = 0
        for f in img_files + mask_files + weather_files:
            total_size += os.path.getsize(f)
        
        # Summary
        print("=" * 80)
        print("✨ DATASET GENERATION COMPLETE!")
        print("=" * 80)
        print()
        print("📊 DATASET STATISTICS:")
        print(f"   ├── Total Samples: {self.num_samples:,}")
        print(f"   ├── Fire Samples (intensity > 0.2): {fire_samples:,} ({100*fire_samples/self.num_samples:.1f}%)")
        print(f"   ├── No-Fire Samples: {no_fire_samples:,} ({100*no_fire_samples/self.num_samples:.1f}%)")
        print(f"   └── Total Size: {total_size / (1024*1024):.1f} MB")
        print()
        print("📁 DATA STRUCTURE:")
        print(f"   ├── {CSV_PATH}")
        print(f"   ├── {IMG_DIR}/ ({len(img_files):,} images)")
        print(f"   ├── {WEATHER_DIR}/ ({len(weather_files):,} .npy files)")
        print(f"   └── {MASK_DIR}/ ({len(mask_files):,} masks)")
        print()
        print("🚀 Dataset ready for training!")
        print("=" * 80)
        
        return fire_df


def main():
    generator = NASAFIRMSDataGenerator()
    generator.generate_complete_dataset()


if __name__ == "__main__":
    main()
