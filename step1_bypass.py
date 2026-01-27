import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# --- LARGE-SCALE DATASET GENERATOR ---
# Generates 10,000+ synthetic fire locations for training a deep learning model
# Real-world fire zones with realistic geographic distribution

def generate_large_dataset(num_samples=10000):
    """
    Generate large-scale synthetic wildfire dataset
    
    Fire hotspots from major regions:
    - California (USA)           - 2500 samples
    - Australia                  - 2500 samples  
    - Brazil (Amazon)            - 2000 samples
    - Indonesia                  - 1500 samples
    - Central Africa             - 1000 samples
    - Mediterranean              - 500 samples
    """
    
    latitudes = []
    longitudes = []
    acq_dates = []
    confidences = []
    
    # Realistic fire location data with geographic clustering
    fire_regions = [
        # California fires (2500 samples) - concentrated around major fire zones
        {
            'name': 'California',
            'lat_center': 38.5,
            'lon_center': -121.0,
            'lat_range': 2.5,
            'lon_range': 3.0,
            'count': 2500,
            'confidence': ['h', 'h', 'h', 'm', 'm']
        },
        # Australia fires (2500 samples) - Blue Mountains and outback
        {
            'name': 'Australia',
            'lat_center': -33.5,
            'lon_center': 150.3,
            'lat_range': 5.0,
            'lon_range': 6.0,
            'count': 2500,
            'confidence': ['h', 'h', 'h', 'm', 'm']
        },
        # Brazil Amazon fires (2000 samples)
        {
            'name': 'Brazil Amazon',
            'lat_center': -10.0,
            'lon_center': -55.0,
            'lat_range': 8.0,
            'lon_range': 10.0,
            'count': 2000,
            'confidence': ['h', 'h', 'm', 'm', 'l']
        },
        # Indonesia fires (1500 samples)
        {
            'name': 'Indonesia',
            'lat_center': 0.0,
            'lon_center': 115.0,
            'lat_range': 8.0,
            'lon_range': 8.0,
            'count': 1500,
            'confidence': ['h', 'h', 'm', 'm', 'l']
        },
        # Central Africa (1000 samples)
        {
            'name': 'Central Africa',
            'lat_center': 0.0,
            'lon_center': 20.0,
            'lat_range': 12.0,
            'lon_range': 15.0,
            'count': 1000,
            'confidence': ['m', 'm', 'm', 'l', 'l']
        },
        # Mediterranean (500 samples)
        {
            'name': 'Mediterranean',
            'lat_center': 40.0,
            'lon_center': 15.0,
            'lat_range': 5.0,
            'lon_range': 30.0,
            'count': 500,
            'confidence': ['h', 'h', 'm', 'm', 'l']
        }
    ]
    
    # Generate dates over 2 years (2022-2024)
    base_date = datetime(2022, 1, 1)
    date_range = 730  # 2 years
    
    print(f"\n📊 Generating {num_samples:,} fire location samples...\n")
    
    sample_idx = 0
    for region in fire_regions:
        print(f"  Generating {region['count']:,} samples for {region['name']}...")
        
        for _ in range(region['count']):
            # Geographic clustering around hotspots with random variation
            lat = region['lat_center'] + np.random.uniform(-region['lat_range']/2, region['lat_range']/2)
            lon = region['lon_center'] + np.random.uniform(-region['lon_range']/2, region['lon_range']/2)
            
            # Random date within range
            random_days = np.random.randint(0, date_range)
            date = base_date + timedelta(days=random_days)
            date_str = date.strftime('%Y-%m-%d')
            
            # Weighted confidence (higher probability of high confidence)
            confidence = np.random.choice(region['confidence'], p=[0.4, 0.25, 0.2, 0.1, 0.05])
            
            latitudes.append(lat)
            longitudes.append(lon)
            acq_dates.append(date_str)
            confidences.append(confidence)
            
            sample_idx += 1
    
    # Create DataFrame
    data = {
        'latitude': latitudes,
        'longitude': longitudes,
        'acq_date': acq_dates,
        'confidence': confidences
    }
    
    df = pd.DataFrame(data)
    
    # Ensure directories exist
    os.makedirs(os.path.join("data", "raw"), exist_ok=True)
    
    output_path = os.path.join("data", "raw", "fire_locations.csv")
    df.to_csv(output_path, index=False)
    
    print("\n" + "="*70)
    print(f"✓ MEGA DATASET GENERATED SUCCESSFULLY!")
    print("="*70)
    print(f"Total Samples: {len(df):,}")
    print(f"Date Range: {df['acq_date'].min()} to {df['acq_date'].max()}")
    print(f"Confidence Breakdown:")
    print(df['confidence'].value_counts().to_string())
    print(f"\nSaved to: {output_path}")
    print(f"\nRegion Distribution:")
    for region in fire_regions:
        print(f"  • {region['name']}: {region['count']:,} samples")
    print("="*70 + "\n")
    print("Next Steps:")
    print("  1. python step2_get_images.py     (Download 10,000+ satellite tiles)")
    print("  2. python step3_process_data.py   (Process weather + create masks)")
    print("  3. python step5_train.py          (Train on 10,000+ samples)")
    print("="*70)
    
    return df

if __name__ == "__main__":
    generate_large_dataset(num_samples=10000)