"""
WildFire AI - Predictive Intelligence System v6.0
Advanced wildfire prediction with Future Presence Mode
"""

import json
import random
import math
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

app = FastAPI(title="WildFire AI Prediction System", version="6.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# DATA MODELS
# ==========================================

class FireLocation(BaseModel):
    lat: float
    lng: float
    hours: Optional[int] = 24

class DroneDeployment(BaseModel):
    lat: float
    lng: float
    count: Optional[int] = 12

class CloudSeedingRequest(BaseModel):
    lat: float
    lng: float

# ==========================================
# PHYSICS-INFORMED PREDICTION ENGINE
# ==========================================

def rothermel_spread_rate(wind_speed_kmh: float, slope_pct: float, fuel_moisture: float) -> float:
    base_rate = 0.5 * (1 - fuel_moisture / 100)
    wind_factor = 1 + (wind_speed_kmh / 20) ** 1.5
    slope_factor = 1 + (slope_pct / 100) * 2 if slope_pct > 0 else 1 / (1 + abs(slope_pct) / 100)
    return base_rate * wind_factor * slope_factor

def calculate_burn_probability(temp_c: float, humidity: float, wind_speed: float, fuel_moisture: float) -> float:
    temp_factor = min(1.0, max(0.0, (temp_c - 10) / 40))
    humidity_factor = 1 - (humidity / 100)
    wind_factor = min(1.0, wind_speed / 60)
    fuel_factor = 1 - (fuel_moisture / 100)
    probability = (temp_factor * 0.25 + humidity_factor * 0.3 + wind_factor * 0.25 + fuel_factor * 0.2) * 100
    return round(min(100, max(0, probability + random.uniform(-5, 5))), 1)

def generate_fire_perimeter(center_lat: float, center_lng: float, hours: int, spread_rate: float, wind_direction: float) -> List[List[float]]:
    base_distance = spread_rate * hours / 111
    perimeter = []
    for angle in range(0, 360, 15):
        rad = math.radians(angle)
        wind_rad = math.radians(wind_direction)
        wind_influence = 0.5 + 0.5 * math.cos(rad - wind_rad)
        distance = base_distance * (0.5 + wind_influence * 0.8) * (0.8 + random.uniform(0, 0.4))
        lat = center_lat + distance * math.cos(rad)
        lng = center_lng + distance * math.sin(rad)
        perimeter.append([lat, lng])
    perimeter.append(perimeter[0])
    return perimeter

def generate_ember_spots(center_lat: float, center_lng: float, wind_direction: float, wind_speed: float) -> List[Dict]:
    spots = []
    wind_rad = math.radians(wind_direction)
    for _ in range(random.randint(3, 7)):
        distance = 0.3 + random.uniform(0, 2) * (wind_speed / 30)
        spread = random.uniform(-0.3, 0.3)
        lat = center_lat + (distance / 111) * math.cos(wind_rad + spread)
        lng = center_lng + (distance / 111) * math.sin(wind_rad + spread)
        spots.append({
            "lat": round(lat, 6),
            "lng": round(lng, 6),
            "probability": round(0.3 + random.uniform(0, 0.6), 2),
            "distance_km": round(distance, 2)
        })
    return spots

# ==========================================
# SENSORY FORECAST GENERATOR
# ==========================================

def generate_sensory_forecast(burn_prob: float, temp: float, wind: float, humidity: float, hours: int) -> List[str]:
    """Generate human-experiential predictions instead of statistics"""
    forecasts = []
    
    if burn_prob > 70:
        forecasts.append("The air will likely sting slightly when you breathe.")
        forecasts.append(f"Visibility may drop enough that distant hills disappear by {hours}:00 PM.")
        forecasts.append("Outdoor noise will be interrupted by aircraft every few minutes.")
        forecasts.append("Ash buildup on flat surfaces becomes noticeable.")
        if temp > 35:
            forecasts.append("Opening windows will let smoke settle on indoor surfaces.")
    elif burn_prob > 50:
        forecasts.append("A faint smell of burning may reach your area by evening.")
        forecasts.append("The sky might take on an orange-gray hue around sunset.")
        forecasts.append("You may notice ash particles on your car windshield.")
        if wind > 25:
            forecasts.append("Gusts may carry visible smoke trails across the horizon.")
    elif burn_prob > 30:
        forecasts.append("Outdoor light may appear slightly hazy tomorrow.")
        forecasts.append("You might notice a faint smoky smell if you're outside for long.")
        if humidity < 30:
            forecasts.append("Dry conditions may make your throat feel scratchy outdoors.")
    else:
        forecasts.append("Conditions appear stable for the next 24 hours.")
        forecasts.append("Air quality is expected to remain within normal range.")
        forecasts.append("No immediate sensory changes anticipated in your area.")
    
    return forecasts[:4]  # Return max 4 forecasts

def generate_future_actions(concern: str) -> Dict:
    """Generate action based on user's selected concern"""
    actions = {
        "air": {
            "action": "Sealing three airflow points before noon reduces indoor smoke exposure tonight.",
            "impact": "Indoor air remains mostly neutral instead of smoky.",
            "timeframe": "6 hours"
        },
        "evacuation": {
            "action": "Pre-loading essentials into your vehicle now saves 45 minutes during an urgent departure.",
            "impact": "You leave calmly instead of rushed.",
            "timeframe": "Immediate"
        },
        "property": {
            "action": "Clearing 30 feet of dry brush from your property line today significantly reduces ember ignition risk.",
            "impact": "Your property becomes defensible space.",
            "timeframe": "4 hours of work"
        },
        "timing": {
            "action": "Setting a departure trigger at 65% burn probability gives you a 3-hour head start.",
            "impact": "You avoid traffic congestion on evacuation routes.",
            "timeframe": "Automatic alert"
        }
    }
    return actions.get(concern, actions["air"])

# ==========================================
# API ENDPOINTS
# ==========================================

@app.get("/api/geocode")
async def geocode_city(city: str = Query(..., description="City name to search")):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://nominatim.openstreetmap.org/search",
                params={"q": city, "format": "json", "limit": 1},
                headers={"User-Agent": "WildFireAI/6.0"}
            )
            data = response.json()
            if data and len(data) > 0:
                result = data[0]
                return {
                    "found": True,
                    "name": result.get("display_name", city),
                    "lat": float(result["lat"]),
                    "lng": float(result["lon"]),
                    "country": result.get("display_name", "").split(",")[-1].strip()
                }
            return {"found": False, "message": f"Location not found: {city}"}
    except Exception as e:
        return {"found": False, "error": str(e)}

@app.get("/api/live-nasa-fires")
async def get_live_fires():
    fires = [
        {"latitude": 34.0522, "longitude": -118.2437, "name": "Los Angeles Basin Fire", "risk_score": 85, "status": "Active", "country": "USA", "brightness": 340, "frp": 45.2},
        {"latitude": 37.7749, "longitude": -122.4194, "name": "San Francisco Wildfire", "risk_score": 72, "status": "Contained", "country": "USA", "brightness": 310, "frp": 28.5},
        {"latitude": -33.8688, "longitude": 151.2093, "name": "Sydney Bush Fire", "risk_score": 91, "status": "Active", "country": "Australia", "brightness": 380, "frp": 67.8},
        {"latitude": 35.6762, "longitude": 139.6503, "name": "Tokyo Forest Fire", "risk_score": 45, "status": "Monitoring", "country": "Japan", "brightness": 290, "frp": 15.2},
        {"latitude": 40.4168, "longitude": -3.7038, "name": "Madrid Wildfire", "risk_score": 67, "status": "Active", "country": "Spain", "brightness": 325, "frp": 35.6},
        {"latitude": -22.9068, "longitude": -43.1729, "name": "Rio Forest Fire", "risk_score": 78, "status": "Active", "country": "Brazil", "brightness": 355, "frp": 52.3},
        {"latitude": 28.6139, "longitude": 77.2090, "name": "Delhi Agricultural Burn", "risk_score": 55, "status": "Monitoring", "country": "India", "brightness": 300, "frp": 22.1},
        {"latitude": 55.7558, "longitude": 37.6173, "name": "Moscow Region Fire", "risk_score": 48, "status": "Contained", "country": "Russia", "brightness": 285, "frp": 18.9},
        {"latitude": -1.2921, "longitude": 36.8219, "name": "Nairobi Brush Fire", "risk_score": 62, "status": "Active", "country": "Kenya", "brightness": 318, "frp": 31.4},
        {"latitude": 31.2304, "longitude": 121.4737, "name": "Shanghai Industrial Fire", "risk_score": 38, "status": "Extinguished", "country": "China", "brightness": 275, "frp": 12.3},
    ]
    for fire in fires:
        fire["risk_score"] = min(100, max(0, fire["risk_score"] + random.randint(-10, 10)))
        fire["detected_time"] = (datetime.now() - timedelta(hours=random.randint(1, 48))).isoformat()
        fire["area_hectares"] = random.randint(50, 5000)
        fire["confidence"] = random.randint(75, 99)
    return {"fires": fires, "count": len(fires), "timestamp": datetime.now().isoformat()}

@app.get("/api/nasa-firms-hotspots")
async def get_nasa_hotspots():
    """Get global wildfire hotspots from NASA FIRMS (simulated for demo)"""
    hotspots = []
    # Generate realistic global distribution of fire hotspots
    regions = [
        {"name": "Amazon Basin", "lat_range": (-10, 5), "lng_range": (-70, -50), "density": 15},
        {"name": "Central Africa", "lat_range": (-10, 10), "lng_range": (15, 35), "density": 20},
        {"name": "Southeast Asia", "lat_range": (5, 20), "lng_range": (95, 115), "density": 12},
        {"name": "Australia", "lat_range": (-35, -20), "lng_range": (120, 150), "density": 10},
        {"name": "Western USA", "lat_range": (32, 45), "lng_range": (-125, -110), "density": 8},
        {"name": "Southern Europe", "lat_range": (35, 45), "lng_range": (-10, 30), "density": 6},
        {"name": "Russia/Siberia", "lat_range": (50, 65), "lng_range": (60, 140), "density": 18},
        {"name": "India", "lat_range": (8, 28), "lng_range": (72, 88), "density": 7},
    ]
    
    for region in regions:
        for _ in range(region["density"]):
            lat = random.uniform(region["lat_range"][0], region["lat_range"][1])
            lng = random.uniform(region["lng_range"][0], region["lng_range"][1])
            hotspots.append({
                "lat": round(lat, 4),
                "lng": round(lng, 4),
                "brightness": random.randint(300, 450),
                "frp": round(random.uniform(10, 100), 1),  # Fire Radiative Power
                "confidence": random.randint(60, 99),
                "satellite": random.choice(["MODIS", "VIIRS", "GOES"]),
                "region": region["name"],
                "scan_time": (datetime.now() - timedelta(hours=random.randint(0, 24))).isoformat()
            })
    
    return {
        "hotspots": hotspots,
        "total_count": len(hotspots),
        "timestamp": datetime.now().isoformat(),
        "source": "NASA FIRMS (Simulated)"
    }

@app.post("/api/predict-fire-spread")
async def predict_fire_spread(request: FireLocation):
    weather = {
        "temperature_c": round(25 + random.uniform(-10, 20), 1),
        "humidity_percent": round(30 + random.uniform(-20, 40), 1),
        "wind_speed_kmh": round(10 + random.uniform(0, 40), 1),
        "wind_direction": random.randint(0, 359),
        "fuel_moisture_percent": round(15 + random.uniform(-10, 30), 1)
    }
    
    burn_prob = calculate_burn_probability(
        weather["temperature_c"],
        weather["humidity_percent"],
        weather["wind_speed_kmh"],
        weather["fuel_moisture_percent"]
    )
    
    spread_rate = rothermel_spread_rate(
        weather["wind_speed_kmh"],
        slope_pct=random.uniform(-10, 20),
        fuel_moisture=weather["fuel_moisture_percent"]
    )
    
    predictions = []
    for hours in [3, 6, 12, 18, 24]:
        if hours <= request.hours:
            perimeter = generate_fire_perimeter(
                request.lat, request.lng, hours,
                spread_rate, weather["wind_direction"]
            )
            # REALISTIC area calculation: spread_rate is km/h, radius = spread_rate * hours
            # But fires don't spread as perfect circles - use ellipse with wind factor
            radius_km = spread_rate * hours * 0.15  # Scale down for realism
            area_sq_km = math.pi * (radius_km ** 2) * 0.7  # Ellipse correction
            area_hectares = area_sq_km * 100  # Convert to hectares
            perimeter_km = 2 * math.pi * radius_km * 0.9  # Perimeter estimate
            predictions.append({
                "hours": hours,
                "perimeter": perimeter,
                "area_hectares": round(min(area_hectares, 500), 1),  # Cap at 500 hectares for 24h
                "area_sq_km": round(min(area_sq_km, 5), 2),
                "burn_probability": min(100, round(burn_prob + hours * 1.5, 1)),
                "perimeter_km": round(min(perimeter_km, 15), 2)  # Cap at 15km
            })
    
    ember_spots = generate_ember_spots(
        request.lat, request.lng,
        weather["wind_direction"],
        weather["wind_speed_kmh"]
    )
    
    # Calculate total predicted spread
    final_prediction = predictions[-1] if predictions else None
    initial_area = 0.5  # Assume 0.5 hectares initial fire (realistic starting point)
    
    # Cap the spread multiplier to realistic values (typically 10-50x over 24h)
    if final_prediction:
        max_realistic_spread = min(final_prediction["area_hectares"], 200)  # Cap at 200 hectares
        final_prediction["area_hectares"] = max_realistic_spread
        final_prediction["perimeter_km"] = min(final_prediction["perimeter_km"], 10)  # Cap at 10km
    
    if burn_prob > 70:
        risk_level = "EXTREME"
    elif burn_prob > 50:
        risk_level = "HIGH"
    elif burn_prob > 30:
        risk_level = "MODERATE"
    else:
        risk_level = "LOW"
    
    # Generate sensory forecasts for Future Presence Mode
    sensory_forecasts = generate_sensory_forecast(
        burn_prob, weather["temperature_c"],
        weather["wind_speed_kmh"], weather["humidity_percent"],
        random.randint(14, 20)
    )
    
    return {
        "burn_probability_score": burn_prob,
        "risk_level": risk_level,
        "spread_rate_kmh": round(spread_rate, 2),
        "weather_factors": weather,
        "predictions": predictions,
        "ember_spots": ember_spots,
        "prediction_summary": {
            "initial_size_hectares": initial_area,
            "predicted_size_24h_hectares": round(min(final_prediction["area_hectares"], 200), 1) if final_prediction else 0,
            "spread_multiplier": round(min(final_prediction["area_hectares"] / initial_area, 100), 1) if final_prediction else 0,
            "predicted_perimeter_km": round(min(final_prediction["perimeter_km"], 10), 1) if final_prediction else 0
        },
        "sensory_forecasts": sensory_forecasts,
        "model": "Rothermel-PINN-v3",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/deploy-drones")
async def deploy_drones(request: DroneDeployment):
    mission_id = f"SWARM-{random.randint(10000, 99999)}"
    base_lat = request.lat + 0.15
    base_lng = request.lng - 0.12
    
    drones = []
    for i in range(request.count):
        angle = (360 / request.count) * i
        rad = math.radians(angle)
        target_lat = request.lat + 0.03 * math.cos(rad)
        target_lng = request.lng + 0.03 * math.sin(rad)
        
        path = []
        for t in range(21):
            progress = t / 20
            mid_lat = (base_lat + target_lat) / 2 + 0.02 * math.sin(progress * math.pi)
            mid_lng = (base_lng + target_lng) / 2
            lat = (1-progress)**2 * base_lat + 2*(1-progress)*progress * mid_lat + progress**2 * target_lat
            lng = (1-progress)**2 * base_lng + 2*(1-progress)*progress * mid_lng + progress**2 * target_lng
            path.append([lat, lng])
        
        drones.append({
            "id": f"DRONE-{i+1:02d}",
            "type": "Fire Suppression UAV",
            "payload": "200L Water + Retardant",
            "path": path,
            "eta_minutes": round(5 + random.uniform(0, 8), 1),
            "battery_percent": random.randint(85, 100),
            "status": "Deployed"
        })
    
    return {
        "mission_id": mission_id,
        "drones": drones,
        "total_deployed": len(drones),
        "estimated_suppression_time_hours": round(2 + random.uniform(0, 4), 1),
        "water_capacity_total_liters": len(drones) * 200,
        "command": "ENGAGE",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/cloud-seeding")
async def initiate_cloud_seeding(request: CloudSeedingRequest):
    operation_id = f"RAIN-{random.randint(1000, 9999)}"
    wind_direction = random.randint(0, 359)
    wind_rad = math.radians(wind_direction)
    
    seeding_points = []
    for i in range(4):
        distance = 0.2 + i * 0.08
        offset = (i - 1.5) * 0.05
        lat = request.lat + distance * math.cos(wind_rad) + offset * math.sin(wind_rad)
        lng = request.lng + distance * math.sin(wind_rad) - offset * math.cos(wind_rad)
        seeding_points.append({
            "id": f"SEED-{i+1}",
            "lat": round(lat, 6),
            "lng": round(lng, 6),
            "altitude_m": random.randint(3000, 5000),
            "agent": "AgI + Liquid Propane"
        })
    
    return {
        "operation_id": operation_id,
        "status": "INITIATED",
        "seeding_points": seeding_points,
        "aircraft_dispatched": 2,
        "expected_precipitation_mm": round(5 + random.uniform(0, 15), 1),
        "eta_rain_hours": round(1.5 + random.uniform(0, 2), 1),
        "coverage_radius_km": round(10 + random.uniform(0, 10), 1),
        "wind_direction_deg": wind_direction,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/future-presence")
async def get_future_presence(request: FireLocation):
    """Generate Future Presence Mode experiential forecast"""
    weather = {
        "temperature_c": round(25 + random.uniform(-10, 20), 1),
        "humidity_percent": round(30 + random.uniform(-20, 40), 1),
        "wind_speed_kmh": round(10 + random.uniform(0, 40), 1),
        "fuel_moisture_percent": round(15 + random.uniform(-10, 30), 1)
    }
    
    burn_prob = calculate_burn_probability(
        weather["temperature_c"],
        weather["humidity_percent"],
        weather["wind_speed_kmh"],
        weather["fuel_moisture_percent"]
    )
    
    forecasts = generate_sensory_forecast(
        burn_prob, weather["temperature_c"],
        weather["wind_speed_kmh"], weather["humidity_percent"],
        random.randint(14, 20)
    )
    
    future_time = datetime.now() + timedelta(hours=request.hours or 24)
    
    return {
        "future_time": future_time.strftime("%A · %I:%M %p"),
        "sensory_forecasts": forecasts,
        "burn_probability": burn_prob,
        "message": "This is what your area is most likely to feel like.",
        "disclaimer": "Not a warning. A preview.",
        "concerns": [
            {"id": "air", "label": "Air inside my home", "icon": "wind"},
            {"id": "evacuation", "label": "Safe evacuation timing", "icon": "route"},
            {"id": "property", "label": "Protecting outdoor property", "icon": "home"},
            {"id": "timing", "label": "Knowing when to leave", "icon": "clock"}
        ]
    }

@app.post("/api/future-action")
async def get_future_action(concern: str = Query("air")):
    """Get specific action based on user's concern"""
    action_data = generate_future_actions(concern)
    return {
        "concern": concern,
        "action": action_data["action"],
        "impact_if_nothing": "By evening, smoke odor reaches indoors.",
        "impact_with_action": action_data["impact"],
        "timeframe": action_data["timeframe"],
        "note": "This does not affect fire behavior — only your experience of it."
    }

@app.get("/api/wildfire-news-trends")
async def get_wildfire_news_trends():
    """Get current wildfire news and trends from multiple sources"""
    # Simulated news data (in production, would integrate with news APIs)
    current_date = datetime.now()
    
    news_items = [
        {
            "id": 1,
            "title": "California Wildfire Season Intensifies with Record Temperatures",
            "source": "Reuters",
            "category": "BREAKING",
            "location": "California, USA",
            "time": (current_date - timedelta(hours=2)).strftime("%I:%M %p"),
            "date": current_date.strftime("%B %d, %Y"),
            "summary": "Record-breaking temperatures across California have led to extreme fire conditions. Multiple new fires reported in Northern California wine country.",
            "affected_area": "45,000 acres",
            "evacuations": "15,000 residents",
            "containment": "25%",
            "image_url": "https://placehold.co/400x200/ff4444/fff?text=California+Fire",
            "urgency": "high"
        },
        {
            "id": 2,
            "title": "Australian Bush Fires Threaten Sydney Suburbs",
            "source": "ABC News Australia",
            "category": "ACTIVE",
            "location": "New South Wales, Australia",
            "time": (current_date - timedelta(hours=5)).strftime("%I:%M %p"),
            "date": current_date.strftime("%B %d, %Y"),
            "summary": "Multiple bushfires burning in the Blue Mountains region are moving towards western Sydney suburbs. Fire services on high alert.",
            "affected_area": "28,000 hectares",
            "evacuations": "8,500 residents",
            "containment": "40%",
            "image_url": "https://placehold.co/400x200/ff6600/fff?text=Australia+Fire",
            "urgency": "high"
        },
        {
            "id": 3,
            "title": "Amazon Rainforest Fire Activity Increases 300%",
            "source": "BBC World",
            "category": "ENVIRONMENTAL",
            "location": "Brazil",
            "time": (current_date - timedelta(hours=8)).strftime("%I:%M %p"),
            "date": current_date.strftime("%B %d, %Y"),
            "summary": "Satellite data shows dramatic increase in fire activity across the Amazon basin. Environmental groups call for immediate action.",
            "affected_area": "120,000 hectares",
            "evacuations": "Indigenous communities displaced",
            "containment": "N/A",
            "image_url": "https://placehold.co/400x200/228B22/fff?text=Amazon+Fires",
            "urgency": "medium"
        },
        {
            "id": 4,
            "title": "Greece Wildfires Force Tourist Evacuations",
            "source": "Euronews",
            "category": "INTERNATIONAL",
            "location": "Rhodes, Greece",
            "time": (current_date - timedelta(hours=12)).strftime("%I:%M %p"),
            "date": current_date.strftime("%B %d, %Y"),
            "summary": "Thousands of tourists evacuated from Greek island as wildfires spread. Navy ships deployed for sea evacuations.",
            "affected_area": "18,000 acres",
            "evacuations": "20,000 tourists + 5,000 residents",
            "containment": "15%",
            "image_url": "https://placehold.co/400x200/1e90ff/fff?text=Greece+Fire",
            "urgency": "high"
        },
        {
            "id": 5,
            "title": "Canada Deploys Military to Fight Wildfires",
            "source": "CBC News",
            "category": "RESPONSE",
            "location": "British Columbia, Canada",
            "time": (current_date - timedelta(hours=18)).strftime("%I:%M %p"),
            "date": current_date.strftime("%B %d, %Y"),
            "summary": "Canadian Armed Forces deployed to assist firefighting efforts. Air quality warnings issued across multiple provinces.",
            "affected_area": "85,000 hectares",
            "evacuations": "12,000 residents",
            "containment": "35%",
            "image_url": "https://placehold.co/400x200/dc143c/fff?text=Canada+Fire",
            "urgency": "medium"
        },
        {
            "id": 6,
            "title": "AI Prediction Systems Show Promise in Early Fire Detection",
            "source": "Nature Climate",
            "category": "TECHNOLOGY",
            "location": "Global",
            "time": (current_date - timedelta(days=1)).strftime("%I:%M %p"),
            "date": (current_date - timedelta(days=1)).strftime("%B %d, %Y"),
            "summary": "New artificial intelligence systems can predict wildfire spread patterns with 94% accuracy, giving communities more time to prepare.",
            "affected_area": "N/A",
            "evacuations": "N/A",
            "containment": "N/A",
            "image_url": "https://placehold.co/400x200/9400d3/fff?text=AI+Technology",
            "urgency": "low"
        }
    ]
    
    # Generate trending statistics
    stats = {
        "active_fires_worldwide": random.randint(350, 450),
        "fires_contained_today": random.randint(15, 30),
        "new_fires_reported": random.randint(20, 40),
        "total_area_burning": f"{random.randint(800, 1200)}K hectares",
        "air_quality_alerts": random.randint(45, 75),
        "evacuation_orders": random.randint(25, 50)
    }
    
    # Regional hotspots
    regional_hotspots = [
        {"region": "Western USA", "severity": "EXTREME", "active_fires": random.randint(40, 60), "trend": "increasing"},
        {"region": "Australia", "severity": "HIGH", "active_fires": random.randint(25, 40), "trend": "stable"},
        {"region": "Amazon Basin", "severity": "HIGH", "active_fires": random.randint(80, 120), "trend": "increasing"},
        {"region": "Mediterranean", "severity": "MODERATE", "active_fires": random.randint(15, 30), "trend": "decreasing"},
        {"region": "Central Africa", "severity": "HIGH", "active_fires": random.randint(100, 150), "trend": "stable"},
        {"region": "Siberia", "severity": "MODERATE", "active_fires": random.randint(30, 50), "trend": "decreasing"}
    ]
    
    return {
        "news": news_items,
        "statistics": stats,
        "regional_hotspots": regional_hotspots,
        "last_updated": current_date.isoformat(),
        "source": "WildFire AI News Aggregator"
    }

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    return html_code_mapbox

# ==========================================
# WILDFIRE AI PREDICTION FRONTEND v6.0
# ==========================================

html_code_mapbox = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WildFire AI | Future Presence Mode</title>
    <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='75' font-size='75'>🔥</text></svg>">
    
    <!-- Leaflet & Draw -->
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet.draw/1.0.4/leaflet.draw.css" />
    <script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet.draw/1.0.4/leaflet.draw.js"></script>
    
    <!-- Icons & Fonts -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@500;700&family=Inter:wght@300;400;500&display=swap" rel="stylesheet">
    
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Rajdhani', sans-serif; background: #0a0a0a; overflow: hidden; display: flex; }
        
        :root {
            --neon-blue: #00f3ff;
            --neon-red: #ff003c;
            --neon-green: #0aff0a;
            --neon-orange: #ff6600;
            --neon-purple: #a855f7;
            --neon-yellow: #ffd700;
            --glass-bg: rgba(8, 12, 20, 0.98);
            --sidebar-width: 35%;
        }
        
        /* SIDEBAR */
        .sidebar {
            width: var(--sidebar-width);
            min-width: 380px;
            max-width: 480px;
            height: 100vh;
            background: var(--glass-bg);
            border-right: 1px solid rgba(0, 243, 255, 0.2);
            display: flex;
            flex-direction: column;
            z-index: 100;
            flex-shrink: 0;
        }
        
        .sidebar-header {
            padding: 18px 22px;
            border-bottom: 1px solid rgba(0, 243, 255, 0.15);
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: linear-gradient(135deg, rgba(0, 50, 80, 0.3), transparent);
        }
        
        .logo-section { display: flex; align-items: center; gap: 14px; }
        
        .logo-icon {
            width: 48px;
            height: 48px;
            background: linear-gradient(135deg, var(--neon-orange), var(--neon-red));
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
            animation: logoPulse 2s infinite;
        }
        
        @keyframes logoPulse {
            0%, 100% { box-shadow: 0 0 15px rgba(255, 100, 0, 0.5); }
            50% { box-shadow: 0 0 30px rgba(255, 100, 0, 0.8); }
        }
        
        .logo-text h1 { font-family: 'Orbitron', monospace; font-size: 1.15rem; color: #fff; letter-spacing: 2px; }
        .logo-text .subtitle { font-size: 0.6rem; color: var(--neon-blue); letter-spacing: 1px; }
        
        .sidebar-content { padding: 18px; flex: 1; overflow-y: auto; }
        .sidebar-content::-webkit-scrollbar { width: 4px; }
        .sidebar-content::-webkit-scrollbar-thumb { background: var(--neon-blue); border-radius: 2px; }
        
        .section-header {
            font-size: 0.65rem;
            color: #555;
            text-transform: uppercase;
            letter-spacing: 2px;
            margin: 22px 0 12px 0;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .section-header::after { content: ''; flex: 1; height: 1px; background: linear-gradient(90deg, rgba(0, 243, 255, 0.3), transparent); }
        
        /* FEATURE INDICATORS - Unique for each */
        .indicator {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 0.55rem;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .indicator-brain { background: rgba(168, 85, 247, 0.2); color: var(--neon-purple); border: 1px solid var(--neon-purple); }
        .indicator-brain::before { content: '🧠'; }
        
        .indicator-fire { background: rgba(255, 0, 60, 0.2); color: var(--neon-red); border: 1px solid var(--neon-red); }
        .indicator-fire::before { content: '🔥'; }
        
        .indicator-satellite { background: rgba(0, 243, 255, 0.2); color: var(--neon-blue); border: 1px solid var(--neon-blue); }
        .indicator-satellite::before { content: '🛰️'; }
        
        .indicator-drone { background: rgba(10, 255, 10, 0.2); color: var(--neon-green); border: 1px solid var(--neon-green); }
        .indicator-drone::before { content: '🚁'; }
        
        .indicator-cloud { background: rgba(100, 150, 255, 0.2); color: #6496ff; border: 1px solid #6496ff; }
        .indicator-cloud::before { content: '☁️'; }
        
        .indicator-future { background: rgba(255, 215, 0, 0.2); color: var(--neon-yellow); border: 1px solid var(--neon-yellow); }
        .indicator-future::before { content: '⏳'; }
        
        .indicator-hotspot { background: rgba(20, 20, 20, 0.8); color: #ff4444; border: 1px solid #ff4444; }
        .indicator-hotspot::before { content: '⬤'; animation: hotspotBlink 1s infinite; }
        
        @keyframes hotspotBlink {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.3; }
        }
        
        /* Buttons */
        .btn {
            width: 100%;
            padding: 14px 18px;
            margin: 8px 0;
            border: 1px solid rgba(0, 243, 255, 0.3);
            background: linear-gradient(135deg, rgba(0, 100, 150, 0.4), rgba(0, 60, 100, 0.2));
            color: #fff;
            font-family: 'Rajdhani', sans-serif;
            font-size: 0.9rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
            cursor: pointer;
            border-radius: 8px;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            position: relative;
        }
        
        .btn:hover { transform: translateY(-2px); box-shadow: 0 5px 25px rgba(0, 243, 255, 0.4); }
        .btn-primary { background: linear-gradient(135deg, rgba(0, 243, 255, 0.4), rgba(0, 150, 200, 0.3)); border-color: var(--neon-blue); }
        .btn-danger { background: linear-gradient(135deg, rgba(255, 0, 60, 0.5), rgba(150, 0, 30, 0.3)); border-color: var(--neon-red); }
        .btn-success { background: linear-gradient(135deg, rgba(10, 255, 10, 0.4), rgba(0, 150, 50, 0.3)); border-color: var(--neon-green); }
        .btn-purple { background: linear-gradient(135deg, rgba(168, 85, 247, 0.4), rgba(100, 50, 150, 0.3)); border-color: var(--neon-purple); }
        .btn-yellow { background: linear-gradient(135deg, rgba(255, 215, 0, 0.4), rgba(200, 150, 0, 0.3)); border-color: var(--neon-yellow); }
        .btn-dark { background: linear-gradient(135deg, rgba(30, 30, 30, 0.9), rgba(10, 10, 10, 0.9)); border-color: #ff4444; }
        
        .btn .indicator { position: absolute; right: 12px; top: 50%; transform: translateY(-50%); }
        
        /* Time Slider */
        .time-slider-container {
            background: linear-gradient(135deg, rgba(168, 85, 247, 0.15), rgba(0, 50, 80, 0.2));
            border: 1px solid rgba(168, 85, 247, 0.3);
            border-radius: 10px;
            padding: 16px;
            margin-bottom: 12px;
        }
        
        .time-slider-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
        .time-slider-header span { font-size: 0.7rem; color: var(--neon-purple); text-transform: uppercase; letter-spacing: 1px; }
        .time-value { font-family: 'Orbitron', monospace; font-size: 1.1rem; color: #fff; background: rgba(168, 85, 247, 0.2); padding: 4px 12px; border-radius: 5px; }
        
        input[type="range"] { width: 100%; -webkit-appearance: none; background: linear-gradient(90deg, var(--neon-green), var(--neon-orange), var(--neon-red)); height: 6px; border-radius: 3px; margin-top: 8px; }
        input[type="range"]::-webkit-slider-thumb { -webkit-appearance: none; width: 22px; height: 22px; background: #fff; border-radius: 50%; cursor: pointer; box-shadow: 0 0 15px var(--neon-purple); border: 3px solid var(--neon-purple); }
        
        /* Search Box */
        .search-box { display: flex; gap: 10px; margin-bottom: 12px; }
        .search-box input { flex: 1; padding: 14px 16px; background: rgba(0, 40, 60, 0.5); border: 1px solid rgba(0, 243, 255, 0.2); border-radius: 8px; color: #fff; font-size: 0.9rem; }
        .search-box input::placeholder { color: #666; }
        .search-box button { padding: 14px 22px; background: var(--neon-blue); border: none; border-radius: 8px; color: #000; font-weight: bold; cursor: pointer; transition: all 0.3s; }
        
        /* Grid Buttons */
        .btn-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin: 15px 0; }
        .grid-btn { padding: 22px 14px; background: rgba(0, 40, 60, 0.35); border: 1px solid rgba(0, 243, 255, 0.25); border-radius: 10px; color: #fff; cursor: pointer; text-align: center; transition: all 0.3s ease; }
        .grid-btn:hover, .grid-btn.active { background: rgba(0, 243, 255, 0.2); border-color: var(--neon-blue); transform: translateY(-2px); }
        .grid-btn i { font-size: 1.5rem; display: block; margin-bottom: 10px; color: var(--neon-blue); }
        .grid-btn span { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1px; }
        
        /* Expandable Sections */
        .expand-section { border: 1px solid rgba(0, 243, 255, 0.15); border-radius: 8px; margin-bottom: 10px; overflow: hidden; }
        .expand-header { padding: 14px 16px; background: rgba(0, 40, 60, 0.25); display: flex; align-items: center; justify-content: space-between; cursor: pointer; font-size: 0.8rem; color: #ccc; text-transform: uppercase; letter-spacing: 1px; transition: all 0.3s; }
        .expand-header:hover { background: rgba(0, 60, 80, 0.4); }
        .expand-header .left { display: flex; align-items: center; gap: 10px; }
        .expand-header i { color: var(--neon-blue); font-size: 1rem; }
        .expand-header .arrow { transition: transform 0.3s; }
        .expand-section.open .expand-header .arrow { transform: rotate(180deg); }
        .expand-content { max-height: 0; overflow: hidden; transition: max-height 0.3s ease; background: rgba(0, 30, 50, 0.3); }
        .expand-section.open .expand-content { max-height: 300px; }
        .expand-inner { padding: 15px; }
        .expand-inner .btn { margin: 5px 0; }
        
        /* Risk Meter */
        .risk-meter { background: rgba(0, 30, 50, 0.4); border: 1px solid rgba(0, 243, 255, 0.2); border-radius: 10px; padding: 18px; margin: 15px 0; }
        .risk-meter-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
        .risk-meter-header span { font-size: 0.7rem; color: #888; text-transform: uppercase; }
        .risk-value { font-family: 'Orbitron', monospace; font-size: 2rem; font-weight: bold; }
        .risk-value.low { color: var(--neon-green); }
        .risk-value.moderate { color: var(--neon-orange); }
        .risk-value.high { color: var(--neon-red); }
        .risk-bar { height: 8px; background: rgba(255, 255, 255, 0.1); border-radius: 4px; overflow: hidden; }
        .risk-bar-fill { height: 100%; border-radius: 4px; transition: width 0.5s, background 0.5s; }
        
        /* Terminal */
        .terminal { background: #000; border-top: 1px solid rgba(0, 243, 255, 0.15); padding: 12px 16px; font-family: 'Courier New', monospace; font-size: 0.7rem; color: var(--neon-green); height: 120px; overflow-y: auto; }
        .terminal::-webkit-scrollbar { width: 3px; }
        .terminal::-webkit-scrollbar-thumb { background: var(--neon-green); }
        .terminal div { margin: 4px 0; opacity: 0.9; }
        .terminal .error { color: var(--neon-red); }
        .terminal .warning { color: var(--neon-orange); }
        .terminal .success { color: var(--neon-green); }
        .terminal .info { color: var(--neon-blue); }
        
        /* MAP AREA */
        .map-area { flex: 1; height: 100vh; position: relative; }
        #leaflet-map { width: 100%; height: 100%; }
        #cesium-globe { width: 100%; height: 100%; display: none; position: absolute; top: 0; left: 0; background: #000; }
        
        /* Drawing Tools */
        .draw-tools { position: absolute; top: 80px; left: 15px; z-index: 1000; display: flex; flex-direction: column; gap: 6px; background: var(--glass-bg); border: 1px solid rgba(0, 243, 255, 0.25); border-radius: 10px; padding: 10px; }
        .draw-tool { width: 42px; height: 42px; background: transparent; border: 1px solid rgba(0, 243, 255, 0.3); border-radius: 8px; color: #fff; font-size: 1rem; cursor: pointer; transition: all 0.2s ease; display: flex; align-items: center; justify-content: center; }
        .draw-tool:hover, .draw-tool.active { background: rgba(0, 243, 255, 0.25); border-color: var(--neon-blue); transform: scale(1.05); }
        
        /* Status Badge */
        .status-badge { position: absolute; top: 15px; right: 15px; background: rgba(10, 255, 10, 0.15); border: 1px solid var(--neon-green); padding: 12px 22px; border-radius: 30px; font-size: 0.8rem; color: var(--neon-green); font-weight: bold; letter-spacing: 1px; z-index: 1000; display: flex; align-items: center; gap: 10px; }
        
        /* Coords */
        .coords-display { position: absolute; bottom: 15px; right: 15px; background: var(--glass-bg); border: 1px solid rgba(0, 243, 255, 0.2); padding: 10px 16px; border-radius: 8px; font-family: 'Orbitron', monospace; font-size: 0.75rem; color: var(--neon-blue); z-index: 1000; }
        
        /* Result Panel */
        .result-panel { position: absolute; top: 70px; right: 15px; width: 340px; max-height: 75vh; overflow-y: auto; background: var(--glass-bg); border: 1px solid rgba(0, 243, 255, 0.3); border-radius: 12px; padding: 20px; z-index: 1000; display: none; }
        .result-panel.visible { display: block; }
        .result-panel h3 { font-family: 'Orbitron', monospace; font-size: 0.85rem; color: var(--neon-blue); margin-bottom: 15px; display: flex; align-items: center; gap: 10px; }
        
        .metric { padding: 12px 14px; margin: 8px 0; background: rgba(0, 50, 80, 0.35); border-left: 4px solid var(--neon-blue); border-radius: 0 8px 8px 0; }
        .metric-label { font-size: 0.6rem; color: #888; text-transform: uppercase; letter-spacing: 1px; }
        .metric-value { font-size: 1.1rem; font-weight: bold; color: var(--neon-blue); margin-top: 4px; }
        .metric.critical { border-left-color: var(--neon-red); }
        .metric.critical .metric-value { color: var(--neon-red); }
        .metric.warning { border-left-color: var(--neon-orange); }
        .metric.warning .metric-value { color: var(--neon-orange); }
        .metric.success { border-left-color: var(--neon-green); }
        .metric.success .metric-value { color: var(--neon-green); }
        
        /* Fire & Drone markers */
        .fire-marker { width: 20px; height: 20px; background: radial-gradient(circle, rgba(255,100,0,0.95), rgba(255,0,0,0.4)); border-radius: 50%; border: 2px solid #ff6600; animation: firePulse 1.5s infinite; }
        @keyframes firePulse { 0%, 100% { transform: scale(1); box-shadow: 0 0 12px rgba(255, 100, 0, 0.7); } 50% { transform: scale(1.25); box-shadow: 0 0 25px rgba(255, 100, 0, 0.9); } }
        .drone-marker { width: 14px; height: 14px; background: var(--neon-blue); border-radius: 50%; border: 2px solid #fff; box-shadow: 0 0 10px var(--neon-blue); }
        
        /* Hotspot marker - dark blooming */
        .hotspot-marker { width: 28px; height: 28px; background: radial-gradient(circle, rgba(255,0,0,0.9) 0%, rgba(255,100,0,0.7) 40%, rgba(0,0,0,0.8) 70%, transparent 100%); border-radius: 50%; border: 2px solid #ff3300; box-shadow: 0 0 15px rgba(255, 0, 0, 0.8), 0 0 30px rgba(0, 0, 0, 0.9); animation: hotspotBloom 2s infinite; }
        .hotspot-circle { animation: hotspotPulse 1.5s infinite; }
        @keyframes hotspotPulse { 0%, 100% { opacity: 0.8; } 50% { opacity: 1; } }
        .leaflet-interactive.hotspot-circle { stroke-width: 2; }
        @keyframes hotspotBloom { 0%, 100% { transform: scale(1); box-shadow: 0 0 20px rgba(0, 0, 0, 0.8), 0 0 40px rgba(100, 0, 0, 0.4); } 50% { transform: scale(1.5); box-shadow: 0 0 35px rgba(0, 0, 0, 0.9), 0 0 60px rgba(150, 0, 0, 0.5); } }
        
        /* Voice Indicator */
        .voice-indicator { position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%); background: var(--glass-bg); border: 1px solid var(--neon-purple); padding: 12px 25px; border-radius: 30px; display: none; align-items: center; gap: 12px; z-index: 2000; }
        .voice-indicator.active { display: flex; }
        .voice-wave { display: flex; gap: 3px; align-items: center; }
        .voice-wave span { width: 4px; height: 20px; background: var(--neon-purple); border-radius: 2px; animation: voiceWave 0.5s infinite ease-in-out; }
        .voice-wave span:nth-child(2) { animation-delay: 0.1s; }
        .voice-wave span:nth-child(3) { animation-delay: 0.2s; }
        .voice-wave span:nth-child(4) { animation-delay: 0.3s; }
        .voice-wave span:nth-child(5) { animation-delay: 0.4s; }
        @keyframes voiceWave { 0%, 100% { height: 8px; } 50% { height: 25px; } }
        
        /* FUTURE PRESENCE MODE OVERLAY */
        .future-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.95);
            z-index: 3000;
            display: none;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            font-family: 'Inter', sans-serif;
            color: #fff;
            padding: 40px;
        }
        
        .future-overlay.visible { display: flex; }
        
        .future-time {
            font-size: 1rem;
            color: rgba(255, 255, 255, 0.5);
            letter-spacing: 2px;
            margin-bottom: 20px;
        }
        
        .future-time .pulse {
            display: inline-block;
            width: 8px;
            height: 8px;
            background: var(--neon-orange);
            border-radius: 50%;
            margin-left: 10px;
            animation: futurePulse 2s infinite;
        }
        
        @keyframes futurePulse {
            0%, 100% { opacity: 0.3; transform: scale(1); }
            50% { opacity: 1; transform: scale(1.2); }
        }
        
        .future-main-text {
            font-size: 1.8rem;
            font-weight: 300;
            text-align: center;
            max-width: 600px;
            line-height: 1.6;
            margin-bottom: 15px;
        }
        
        .future-sub-text {
            font-size: 0.85rem;
            color: rgba(255, 255, 255, 0.4);
            margin-bottom: 40px;
        }
        
        .future-forecasts {
            max-width: 500px;
            text-align: left;
        }
        
        .future-forecast-item {
            font-size: 1.1rem;
            color: rgba(255, 255, 255, 0.8);
            margin: 20px 0;
            padding-left: 20px;
            border-left: 2px solid rgba(255, 150, 0, 0.5);
            opacity: 0;
            animation: forecastFadeIn 0.8s ease forwards;
        }
        
        .future-forecast-item:nth-child(1) { animation-delay: 0.5s; }
        .future-forecast-item:nth-child(2) { animation-delay: 1.2s; }
        .future-forecast-item:nth-child(3) { animation-delay: 1.9s; }
        .future-forecast-item:nth-child(4) { animation-delay: 2.6s; }
        
        @keyframes forecastFadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .future-footer {
            margin-top: 50px;
            font-size: 0.75rem;
            color: rgba(255, 255, 255, 0.3);
        }
        
        .future-btn {
            margin-top: 30px;
            padding: 16px 40px;
            background: transparent;
            border: 1px solid rgba(255, 255, 255, 0.3);
            color: #fff;
            font-size: 0.9rem;
            cursor: pointer;
            border-radius: 30px;
            transition: all 0.3s;
        }
        
        .future-btn:hover {
            background: rgba(255, 255, 255, 0.1);
            border-color: var(--neon-orange);
        }
        
        .future-close {
            position: absolute;
            top: 30px;
            right: 30px;
            background: none;
            border: none;
            color: rgba(255, 255, 255, 0.5);
            font-size: 1.5rem;
            cursor: pointer;
        }
        
        /* NEWS TRENDS OVERLAY */
        .news-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(135deg, #0d0d1a 0%, #1a1a2e 50%, #16213e 100%);
            z-index: 4000;
            display: none;
            flex-direction: column;
            overflow-y: auto;
            font-family: 'Inter', sans-serif;
        }
        
        .news-overlay.visible { display: flex; }
        
        .news-header {
            background: linear-gradient(90deg, rgba(233, 69, 96, 0.2), rgba(15, 52, 96, 0.3));
            padding: 25px 40px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid rgba(233, 69, 96, 0.3);
            position: sticky;
            top: 0;
            z-index: 10;
            backdrop-filter: blur(10px);
        }
        
        .news-title {
            display: flex;
            align-items: center;
            gap: 15px;
        }
        
        .news-title h1 {
            font-size: 1.8rem;
            font-weight: 700;
            color: #fff;
            margin: 0;
        }
        
        .news-title .fire-icon {
            font-size: 2rem;
            animation: firePulse 1.5s infinite;
        }
        
        .news-close {
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            color: #fff;
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 0.9rem;
            transition: all 0.3s;
        }
        
        .news-close:hover { background: rgba(255, 255, 255, 0.2); }
        
        .news-stats {
            display: grid;
            grid-template-columns: repeat(6, 1fr);
            gap: 15px;
            padding: 25px 40px;
            background: rgba(0, 0, 0, 0.3);
        }
        
        .stat-card {
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0.02));
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
        }
        
        .stat-card .value {
            font-size: 2rem;
            font-weight: 700;
            color: var(--neon-orange);
            font-family: 'Orbitron', monospace;
        }
        
        .stat-card .label {
            font-size: 0.7rem;
            color: rgba(255, 255, 255, 0.5);
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-top: 8px;
        }
        
        .news-content {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 30px;
            padding: 30px 40px;
        }
        
        .news-feed h2 {
            font-size: 1.2rem;
            color: rgba(255, 255, 255, 0.8);
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .news-card {
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.08), rgba(255, 255, 255, 0.02));
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 20px;
            transition: all 0.3s;
        }
        
        .news-card:hover {
            transform: translateY(-3px);
            border-color: rgba(233, 69, 96, 0.4);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        }
        
        .news-card.breaking { border-left: 4px solid #e94560; }
        .news-card.active { border-left: 4px solid #ff6b6b; }
        .news-card.environmental { border-left: 4px solid #4ecdc4; }
        .news-card.technology { border-left: 4px solid #9400d3; }
        
        .news-card-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 12px;
        }
        
        .news-category {
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 0.65rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .news-category.breaking { background: #e94560; color: #fff; }
        .news-category.active { background: #ff6b6b; color: #fff; }
        .news-category.environmental { background: #4ecdc4; color: #000; }
        .news-category.international { background: #1e90ff; color: #fff; }
        .news-category.response { background: #dc143c; color: #fff; }
        .news-category.technology { background: #9400d3; color: #fff; }
        
        .news-source {
            font-size: 0.7rem;
            color: rgba(255, 255, 255, 0.4);
        }
        
        .news-card h3 {
            font-size: 1.1rem;
            font-weight: 600;
            color: #fff;
            margin-bottom: 10px;
            line-height: 1.4;
        }
        
        .news-card p {
            font-size: 0.85rem;
            color: rgba(255, 255, 255, 0.6);
            line-height: 1.6;
            margin-bottom: 15px;
        }
        
        .news-meta {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
            font-size: 0.75rem;
            padding-top: 15px;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .news-meta-item {
            color: rgba(255, 255, 255, 0.5);
        }
        
        .news-meta-item strong {
            color: var(--neon-orange);
            display: block;
        }
        
        .regional-hotspots h2 {
            font-size: 1.2rem;
            color: rgba(255, 255, 255, 0.8);
            margin-bottom: 20px;
        }
        
        .hotspot-card {
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0.02));
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 12px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .hotspot-card .region { font-weight: 600; color: #fff; }
        .hotspot-card .fires { font-size: 0.85rem; color: rgba(255, 255, 255, 0.5); }
        
        .severity-badge {
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 0.65rem;
            font-weight: 700;
        }
        
        .severity-badge.extreme { background: #ff0000; color: #fff; }
        .severity-badge.high { background: #ff6b00; color: #fff; }
        .severity-badge.moderate { background: #ffd700; color: #000; }
        
        .trend-indicator {
            font-size: 0.75rem;
            display: flex;
            align-items: center;
            gap: 5px;
        }
        
        .trend-indicator.increasing { color: #ff4444; }
        .trend-indicator.stable { color: #ffd700; }
        .trend-indicator.decreasing { color: #44ff44; }
        
        /* Hide defaults */
        .leaflet-control-attribution { display: none !important; }
        .cesium-credit-logoContainer, .cesium-credit-textContainer, .cesium-viewer-bottom { display: none !important; }
        
        /* Loading indicator */
        .loading-3d {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: var(--neon-blue);
            font-size: 1rem;
            z-index: 1001;
            display: none;
        }
        
        .loading-3d.visible { display: block; }
        
        @media (max-width: 1200px) {
            .news-stats { grid-template-columns: repeat(3, 1fr); }
            .news-content { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <!-- SIDEBAR -->
    <div class="sidebar">
        <div class="sidebar-header">
            <div class="logo-section">
                <div class="logo-icon">🔥</div>
                <div class="logo-text">
                    <h1>WILDFIRE AI</h1>
                    <div class="subtitle">FUTURE PRESENCE MODE</div>
                </div>
            </div>
        </div>
        
        <div class="sidebar-content">
            <!-- ANALYZE FIRE SPREAD - TOP -->
            <button class="btn btn-primary" onclick="runPrediction()" style="position: relative;">
                <i class="fas fa-brain"></i> ANALYZE FIRE SPREAD
                <span class="indicator indicator-brain">AI</span>
            </button>
            
            <!-- Predictive Timeline -->
            <div class="time-slider-container">
                <div class="time-slider-header">
                    <span><i class="fas fa-clock"></i> Predictive Timeline</span>
                    <div class="time-value" id="timeValue">NOW</div>
                </div>
                <input type="range" min="0" max="24" value="0" id="timeSlider" oninput="updatePrediction(this.value)">
                <div style="display: flex; justify-content: space-between; font-size: 0.6rem; color: #666; margin-top: 6px;">
                    <span>NOW</span><span>+6H</span><span>+12H</span><span>+18H</span><span>+24H</span>
                </div>
            </div>
            
            <!-- Go To Live Wildfire -->
            <button class="btn btn-danger" onclick="goToLiveWildfire()" style="position: relative;">
                <i class="fas fa-fire"></i> GO TO LIVE WILDFIRE
                <span class="indicator indicator-fire">LIVE</span>
            </button>
            
            <!-- NASA FIRMS Hotspots -->
            <button class="btn btn-dark" onclick="showWildfireHotspots()" style="position: relative;">
                <i class="fas fa-globe-americas"></i> WILDFIRE HOTSPOTS
                <span class="indicator indicator-hotspot">NASA</span>
            </button>
            
            <!-- Wildfire News Trends -->
            <button class="btn btn-news" onclick="showWildfireNewsTrends()" style="position: relative; background: linear-gradient(135deg, #1a1a2e, #16213e); border: 1px solid #0f3460;">
                <i class="fas fa-newspaper"></i> WILDFIRE TRENDS
                <span class="indicator" style="background: linear-gradient(90deg, #e94560, #ff6b6b); animation: pulse 1.5s infinite;">NEWS</span>
            </button>
            
            <div class="section-header">Location Search</div>
            <div class="search-box">
                <input type="text" id="citySearch" placeholder="Search any city worldwide..." onkeypress="if(event.key==='Enter')searchCity()">
                <button onclick="searchCity()">GO</button>
            </div>
            
            <!-- View Mode Grid -->
            <div class="btn-grid">
                <div class="grid-btn" onclick="toggleSatellite()" id="btnMapType">
                    <i class="fas fa-map"></i>
                    <span>Map Type</span>
                </div>
                <div class="grid-btn" id="btn3D" onclick="toggle3DGlobe()">
                    <i class="fas fa-cube"></i>
                    <span>3D Holo</span>
                </div>
            </div>
            
            <!-- Risk Meter -->
            <div class="risk-meter">
                <div class="risk-meter-header">
                    <span>Burn Probability Score</span>
                    <span class="risk-value low" id="riskValue">--</span>
                </div>
                <div class="risk-bar">
                    <div class="risk-bar-fill" id="riskBarFill" style="width: 0%; background: var(--neon-green);"></div>
                </div>
            </div>
            
            <div class="section-header">Response Actions</div>
            
            <!-- Drone Swarm -->
            <div class="expand-section" id="droneSection">
                <div class="expand-header" onclick="toggleExpand('droneSection')">
                    <div class="left">
                        <i class="fas fa-helicopter"></i>
                        <span>Autonomous Drone Swarm</span>
                        <span class="indicator indicator-drone" style="margin-left: 10px;">12 Units</span>
                    </div>
                    <i class="fas fa-chevron-down arrow"></i>
                </div>
                <div class="expand-content">
                    <div class="expand-inner">
                        <p style="font-size: 0.75rem; color: #888; margin-bottom: 12px;">Deploy AI-controlled drone swarm for fire suppression</p>
                        <button class="btn btn-success" onclick="deployDrones()">
                            <i class="fas fa-play"></i> DEPLOY DRONES
                        </button>
                    </div>
                </div>
            </div>
            
            <!-- Cloud Seeding -->
            <div class="expand-section" id="cloudSection">
                <div class="expand-header" onclick="toggleExpand('cloudSection')">
                    <div class="left">
                        <i class="fas fa-cloud-rain"></i>
                        <span>Cloud Seeding</span>
                        <span class="indicator indicator-cloud" style="margin-left: 10px;">AgI</span>
                    </div>
                    <i class="fas fa-chevron-down arrow"></i>
                </div>
                <div class="expand-content">
                    <div class="expand-inner">
                        <button class="btn btn-purple" onclick="initiateCloudSeeding()">
                            <i class="fas fa-cloud"></i> START CLOUD SEEDING
                        </button>
                    </div>
                </div>
            </div>
            
            <div class="section-header">Advanced Features</div>
            
            <!-- Future Presence Mode -->
            <button class="btn btn-yellow" onclick="enterFuturePresence()" style="position: relative;">
                <i class="fas fa-eye"></i> FUTURE PRESENCE MODE
                <span class="indicator indicator-future">NEW</span>
            </button>
            
            <!-- Ember-Cast -->
            <button class="btn" onclick="runEmberSimulation()" style="position: relative;">
                <i class="fas fa-wind"></i> EMBER-CAST TRAJECTORY
            </button>
            
            <!-- Voice Assistant -->
            <button class="btn btn-purple" onclick="toggleVoiceAssistant()">
                <i class="fas fa-microphone"></i> AI VOICE ASSISTANT
            </button>
        </div>
        
        <!-- Terminal -->
        <div class="terminal" id="terminal">
            <div class="info">> WILDFIRE AI v6.0 INITIALIZED</div>
            <div>> FUTURE PRESENCE MODE: READY</div>
            <div>> NASA FIRMS HOTSPOT LINK: ACTIVE</div>
            <div class="success">> ALL SYSTEMS OPERATIONAL</div>
        </div>
    </div>
    
    <!-- MAP AREA -->
    <div class="map-area">
        <div id="leaflet-map"></div>
        <div id="cesium-globe"></div>
        <div class="loading-3d" id="loading3D"><i class="fas fa-spinner fa-spin"></i> Loading 3D Globe...</div>
        
        <!-- Drawing Tools -->
        <div class="draw-tools" id="drawTools">
            <button class="draw-tool" onclick="startDraw('polygon')" title="Draw Polygon"><i class="fas fa-draw-polygon"></i></button>
            <button class="draw-tool" onclick="startDraw('rectangle')" title="Rectangle"><i class="fas fa-square"></i></button>
            <button class="draw-tool" onclick="startDraw('circle')" title="Circle"><i class="fas fa-circle"></i></button>
            <button class="draw-tool" onclick="clearDrawings()" title="Clear All"><i class="fas fa-trash"></i></button>
        </div>
        
        <!-- Status Badge -->
        <div class="status-badge">
            <i class="fas fa-satellite"></i> NASA FIRMS LINKED
        </div>
        
        <!-- Coords -->
        <div class="coords-display" id="coords">LAT: 0.0000 | LNG: 0.0000</div>
        
        <!-- Result Panel -->
        <div class="result-panel" id="resultPanel">
            <h3><i class="fas fa-chart-line"></i> PREDICTION RESULTS</h3>
            <div id="resultsContent"></div>
            <button class="btn btn-danger" style="margin-top: 15px;" onclick="closeResults()">
                <i class="fas fa-times"></i> Close
            </button>
        </div>
    </div>
    
    <!-- Voice Indicator -->
    <div class="voice-indicator" id="voiceIndicator">
        <div class="voice-wave"><span></span><span></span><span></span><span></span><span></span></div>
        <span style="color: var(--neon-purple); font-size: 0.85rem;">AI Assistant Speaking...</span>
    </div>
    
    <!-- Future Presence Overlay -->
    <div class="future-overlay" id="futureOverlay">
        <button class="future-close" onclick="closeFuturePresence()">&times;</button>
        <div class="future-time" id="futureTime">Tomorrow · 4:00 PM <span class="pulse"></span></div>
        <div class="future-main-text">"This is what your area is most likely to feel like."</div>
        <div class="future-sub-text">Not a warning. A preview.</div>
        <div class="future-forecasts" id="futureForecastsContainer"></div>
        <div class="future-footer">This is based on how similar conditions have unfolded before.</div>
        <button class="future-btn" onclick="closeFuturePresence()">Return to Present</button>
    </div>
    
    <!-- News Trends Overlay -->
    <div class="news-overlay" id="newsOverlay">
        <div class="news-header">
            <div class="news-title">
                <span class="fire-icon">🔥</span>
                <h1>WILDFIRE GLOBAL TRENDS</h1>
                <span style="padding: 5px 15px; background: #e94560; border-radius: 5px; font-size: 0.7rem; font-weight: 700;">LIVE</span>
            </div>
            <button class="news-close" onclick="closeNewsTrends()">
                <i class="fas fa-times"></i> Close Dashboard
            </button>
        </div>
        
        <div class="news-stats" id="newsStats">
            <!-- Stats populated by JS -->
        </div>
        
        <div class="news-content">
            <div class="news-feed">
                <h2><i class="fas fa-rss"></i> Latest Wildfire Reports</h2>
                <div id="newsFeedContainer">
                    <!-- News cards populated by JS -->
                </div>
            </div>
            
            <div class="regional-hotspots">
                <h2><i class="fas fa-map-marked-alt"></i> Regional Hotspots</h2>
                <div id="regionalHotspotsContainer">
                    <!-- Hotspot cards populated by JS -->
                </div>
            </div>
        </div>
    </div>
    
    <!-- Load CesiumJS after page load for faster initial render -->
    <script>
        // Pre-load Cesium
        let cesiumLoaded = false;
        let cesiumScript = null;
        let cesiumCSS = null;
        
        function preloadCesium() {
            if (cesiumLoaded) return Promise.resolve();
            
            return new Promise((resolve) => {
                // Load CSS first
                cesiumCSS = document.createElement('link');
                cesiumCSS.rel = 'stylesheet';
                cesiumCSS.href = 'https://cesium.com/downloads/cesiumjs/releases/1.95/Build/Cesium/Widgets/widgets.css';
                document.head.appendChild(cesiumCSS);
                
                // Load JS
                cesiumScript = document.createElement('script');
                cesiumScript.src = 'https://cesium.com/downloads/cesiumjs/releases/1.95/Build/Cesium/Cesium.js';
                cesiumScript.onload = () => {
                    cesiumLoaded = true;
                    resolve();
                };
                document.head.appendChild(cesiumScript);
            });
        }
        
        // Start preloading Cesium immediately
        preloadCesium();
    </script>
    
    <script>
        // =============================================
        // INITIALIZATION
        // =============================================
        
        const satelliteLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {});
        const streetLayer = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {});
        
        const map = L.map('leaflet-map', {
            center: [20, 0],
            zoom: 3,
            layers: [satelliteLayer],
            zoomControl: true
        });
        
        let isSatellite = true;
        let is3DMode = false;
        let cesiumViewer = null;
        let cesiumEntities = [];
        let fireMarkers = [];
        let hotspotMarkers = [];
        let activeFires = [];
        let predictionPolygons = [];
        let droneMarkers = [];
        let currentPredictionHours = 0;
        let lastPredictionData = null;
        let voiceEnabled = false;
        let selectedFireLocation = null;
        
        // Leaflet Drawing
        const drawnItems = new L.FeatureGroup();
        map.addLayer(drawnItems);
        let currentDrawer = null;
        
        // Voice Synthesis
        const synth = window.speechSynthesis;
        
        // =============================================
        // VOICE ASSISTANT
        // =============================================
        
        function speak(text) {
            if (!voiceEnabled) return;
            synth.cancel();
            const utterance = new SpeechSynthesisUtterance(text);
            const voices = synth.getVoices();
            const femaleVoice = voices.find(v => v.name.includes('Female') || v.name.includes('Zira') || v.name.includes('Samantha')) || voices[0];
            utterance.voice = femaleVoice;
            utterance.rate = 1.0;
            utterance.pitch = 1.1;
            utterance.onstart = () => document.getElementById('voiceIndicator').classList.add('active');
            utterance.onend = () => document.getElementById('voiceIndicator').classList.remove('active');
            synth.speak(utterance);
        }
        
        function toggleVoiceAssistant() {
            voiceEnabled = !voiceEnabled;
            if (voiceEnabled) {
                log('AI Voice Assistant: ENABLED', 'success');
                speak('Voice assistant activated.');
            } else {
                synth.cancel();
                document.getElementById('voiceIndicator').classList.remove('active');
                log('AI Voice Assistant: DISABLED', 'warning');
            }
        }
        
        // =============================================
        // DRAWING FUNCTIONS - Works in both 2D and 3D
        // =============================================
        
        function startDraw(type) {
            if (is3DMode) {
                // 3D drawing mode
                start3DDraw(type);
            } else {
                // 2D Leaflet drawing
                if (currentDrawer) currentDrawer.disable();
                const options = { shapeOptions: { color: '#00f3ff', fillColor: '#00f3ff', fillOpacity: 0.3, weight: 2 } };
                if (type === 'polygon') currentDrawer = new L.Draw.Polygon(map, options);
                else if (type === 'rectangle') currentDrawer = new L.Draw.Rectangle(map, options);
                else if (type === 'circle') currentDrawer = new L.Draw.Circle(map, options);
                currentDrawer.enable();
                log('Draw ' + type + ' on map');
            }
        }
        
        let cesiumDrawHandler = null;
        let cesiumDrawPositions = [];
        
        function start3DDraw(type) {
            if (!cesiumViewer) return;
            
            // Clear previous
            cesiumDrawPositions = [];
            
            log('3D Drawing: Click on globe to place points, double-click to finish', 'info');
            
            cesiumDrawHandler = new Cesium.ScreenSpaceEventHandler(cesiumViewer.canvas);
            
            cesiumDrawHandler.setInputAction(function(click) {
                const cartesian = cesiumViewer.camera.pickEllipsoid(click.position, cesiumViewer.scene.globe.ellipsoid);
                if (cartesian) {
                    cesiumDrawPositions.push(cartesian);
                    
                    // Add point marker
                    cesiumViewer.entities.add({
                        position: cartesian,
                        point: { pixelSize: 10, color: Cesium.Color.CYAN }
                    });
                    
                    if (cesiumDrawPositions.length > 1) {
                        // Draw line
                        cesiumViewer.entities.add({
                            polyline: {
                                positions: cesiumDrawPositions.slice(-2),
                                width: 3,
                                material: Cesium.Color.CYAN.withAlpha(0.7)
                            }
                        });
                    }
                    
                    // Update selected location
                    const carto = Cesium.Cartographic.fromCartesian(cartesian);
                    selectedFireLocation = {
                        lat: Cesium.Math.toDegrees(carto.latitude),
                        lng: Cesium.Math.toDegrees(carto.longitude)
                    };
                }
            }, Cesium.ScreenSpaceEventType.LEFT_CLICK);
            
            cesiumDrawHandler.setInputAction(function() {
                if (cesiumDrawPositions.length > 2) {
                    // Complete polygon
                    cesiumViewer.entities.add({
                        polygon: {
                            hierarchy: cesiumDrawPositions,
                            material: Cesium.Color.CYAN.withAlpha(0.3),
                            outline: true,
                            outlineColor: Cesium.Color.CYAN
                        }
                    });
                    cesiumEntities.push(...cesiumViewer.entities.values);
                }
                cesiumDrawHandler.destroy();
                cesiumDrawHandler = null;
                log('3D Area selected', 'success');
            }, Cesium.ScreenSpaceEventType.LEFT_DOUBLE_CLICK);
        }
        
        function clearDrawings() {
            drawnItems.clearLayers();
            predictionPolygons.forEach(p => map.removeLayer(p));
            predictionPolygons = [];
            droneMarkers.forEach(m => map.removeLayer(m));
            droneMarkers = [];
            hotspotMarkers.forEach(m => map.removeLayer(m));
            hotspotMarkers = [];
            
            // Clear 3D entities
            if (cesiumViewer) {
                cesiumViewer.entities.removeAll();
                cesiumEntities = [];
            }
            
            updateRiskMeter(0);
            log('All drawings cleared');
        }
        
        map.on(L.Draw.Event.CREATED, function(e) {
            drawnItems.addLayer(e.layer);
            let center;
            if (e.layer.getBounds) center = e.layer.getBounds().getCenter();
            else if (e.layer.getLatLng) center = e.layer.getLatLng();
            if (center) {
                selectedFireLocation = { lat: center.lat, lng: center.lng };
                log('Area selected at: ' + center.lat.toFixed(4) + ', ' + center.lng.toFixed(4), 'info');
            }
        });
        
        // =============================================
        // PREDICTION FUNCTIONS
        // =============================================
        
        function updatePrediction(hours) {
            currentPredictionHours = parseInt(hours);
            document.getElementById('timeValue').textContent = hours == 0 ? 'NOW' : '+' + hours + 'H';
            if (lastPredictionData && lastPredictionData.predictions) {
                displayPredictionForHour(hours);
            }
        }
        
        function displayPredictionForHour(hours) {
            if (!lastPredictionData) return;
            
            // Clear existing
            predictionPolygons.forEach(p => {
                if (map.hasLayer(p)) map.removeLayer(p);
            });
            predictionPolygons = [];
            
            // Clear 3D predictions
            if (is3DMode && cesiumViewer) {
                cesiumViewer.entities.removeAll();
            }
            
            const prediction = lastPredictionData.predictions.find(p => p.hours >= hours) || 
                               lastPredictionData.predictions[lastPredictionData.predictions.length - 1];
            
            if (prediction && prediction.perimeter) {
                const color = prediction.burn_probability > 70 ? '#ff003c' : 
                              prediction.burn_probability > 40 ? '#ff6600' : '#0aff0a';
                
                if (is3DMode && cesiumViewer) {
                    // 3D polygon
                    const positions = prediction.perimeter.map(p => Cesium.Cartesian3.fromDegrees(p[1], p[0]));
                    cesiumViewer.entities.add({
                        polygon: {
                            hierarchy: positions,
                            material: Cesium.Color.fromCssColorString(color).withAlpha(0.4),
                            outline: true,
                            outlineColor: Cesium.Color.fromCssColorString(color),
                            outlineWidth: 3
                        }
                    });
                } else {
                    // 2D polygon
                    const polygon = L.polygon(prediction.perimeter, {
                        color: color,
                        fillColor: color,
                        fillOpacity: 0.35,
                        weight: 3,
                        dashArray: '10, 5'
                    }).addTo(map);
                    
                    polygon.bindPopup(
                        '<b style="color:' + color + ';">T+' + prediction.hours + 'H Prediction</b><br>' +
                        'Burn Probability: ' + prediction.burn_probability + '%<br>' +
                        'Area: ' + prediction.area_hectares + ' hectares<br>' +
                        'Perimeter: ' + prediction.perimeter_km + ' km'
                    );
                    predictionPolygons.push(polygon);
                }
                
                updateRiskMeter(prediction.burn_probability);
            }
        }
        
        async function runPrediction() {
            if (!selectedFireLocation) {
                // Use map center if no selection
                if (is3DMode && cesiumViewer) {
                    const center = cesiumViewer.camera.positionCartographic;
                    selectedFireLocation = {
                        lat: Cesium.Math.toDegrees(center.latitude),
                        lng: Cesium.Math.toDegrees(center.longitude)
                    };
                } else {
                    const center = map.getCenter();
                    selectedFireLocation = { lat: center.lat, lng: center.lng };
                }
            }
            
            log('Running Physics-Informed Fire Spread Analysis...', 'info');
            speak('Analyzing fire spread using physics-informed neural network.');
            
            try {
                const response = await fetch('/api/predict-fire-spread', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ lat: selectedFireLocation.lat, lng: selectedFireLocation.lng, hours: 24 })
                });
                
                const data = await response.json();
                lastPredictionData = data;
                
                displayPredictionResults(data);
                displayPredictionForHour(currentPredictionHours || 6);
                
                if (data.ember_spots && data.ember_spots.length > 0) {
                    displayEmberSpots(data.ember_spots);
                }
                
                log('Prediction complete: ' + data.burn_probability_score + '% burn probability', 'success');
                speak('Analysis complete. Burn probability is ' + Math.round(data.burn_probability_score) + ' percent.');
                
            } catch (e) {
                log('Prediction error: ' + e.message, 'error');
            }
        }
        
        function displayPredictionResults(data) {
            const riskClass = data.burn_probability_score > 70 ? 'critical' : data.burn_probability_score > 40 ? 'warning' : '';
            const summary = data.prediction_summary || {};
            
            let html = `
                <div class="metric ${riskClass}">
                    <div class="metric-label">Burn Probability Score</div>
                    <div class="metric-value">${data.burn_probability_score}%</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Risk Level</div>
                    <div class="metric-value">${data.risk_level}</div>
                </div>
                <div class="metric success">
                    <div class="metric-label">Initial Fire Size</div>
                    <div class="metric-value">${summary.initial_size_hectares || 0.5} hectares</div>
                </div>
                <div class="metric ${riskClass}">
                    <div class="metric-label">Predicted Size (24h)</div>
                    <div class="metric-value">${summary.predicted_size_24h_hectares || 0} hectares</div>
                </div>
                <div class="metric warning">
                    <div class="metric-label">Fire Spread Multiplier</div>
                    <div class="metric-value">${summary.spread_multiplier || 0}x expansion</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Predicted Perimeter</div>
                    <div class="metric-value">${summary.predicted_perimeter_km || 0} km</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Spread Rate</div>
                    <div class="metric-value">${data.spread_rate_kmh} km/h</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Temperature</div>
                    <div class="metric-value">${data.weather_factors.temperature_c}°C</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Wind Speed</div>
                    <div class="metric-value">${data.weather_factors.wind_speed_kmh} km/h</div>
                </div>
            `;
            
            if (data.ember_spots && data.ember_spots.length > 0) {
                html += `<div class="metric warning"><div class="metric-label">Ember-Cast Alert</div><div class="metric-value">${data.ember_spots.length} Spot Fire Risks</div></div>`;
            }
            
            document.getElementById('resultsContent').innerHTML = html;
            document.getElementById('resultPanel').classList.add('visible');
        }
        
        function displayEmberSpots(spots) {
            spots.forEach(spot => {
                if (is3DMode && cesiumViewer) {
                    cesiumViewer.entities.add({
                        position: Cesium.Cartesian3.fromDegrees(spot.lng, spot.lat),
                        point: { pixelSize: 15, color: Cesium.Color.ORANGE, outlineColor: Cesium.Color.WHITE, outlineWidth: 2 }
                    });
                } else {
                    const marker = L.circleMarker([spot.lat, spot.lng], {
                        radius: 10, color: '#ff9500', fillColor: '#ff9500', fillOpacity: 0.6, weight: 2
                    }).addTo(map);
                    marker.bindPopup('<b style="color:#ff9500;">⚡ Ember Spot</b><br>Probability: ' + (spot.probability * 100).toFixed(0) + '%');
                    predictionPolygons.push(marker);
                }
            });
        }
        
        function updateRiskMeter(value) {
            const riskValue = document.getElementById('riskValue');
            const riskBarFill = document.getElementById('riskBarFill');
            riskValue.textContent = value > 0 ? value + '%' : '--';
            riskBarFill.style.width = value + '%';
            if (value > 70) { riskValue.className = 'risk-value high'; riskBarFill.style.background = 'linear-gradient(90deg, #ff6600, #ff003c)'; }
            else if (value > 40) { riskValue.className = 'risk-value moderate'; riskBarFill.style.background = 'linear-gradient(90deg, #ffcc00, #ff6600)'; }
            else { riskValue.className = 'risk-value low'; riskBarFill.style.background = 'linear-gradient(90deg, #0aff0a, #00cc00)'; }
        }
        
        function closeResults() { document.getElementById('resultPanel').classList.remove('visible'); }
        
        // =============================================
        // NASA FIRMS HOTSPOTS - Dark Blooming Animation
        // =============================================
        
        async function showWildfireHotspots() {
            log('Fetching NASA FIRMS hotspot data...', 'info');
            speak('Loading global wildfire hotspots from NASA FIRMS satellite data.');
            
            try {
                const response = await fetch('/api/nasa-firms-hotspots');
                const data = await response.json();
                
                // Clear old hotspots from 2D map
                hotspotMarkers.forEach(m => {
                    try { map.removeLayer(m); } catch(e) {}
                });
                hotspotMarkers = [];
                
                // Clear old 3D entities if in 3D mode
                if (cesiumViewer) {
                    cesiumEntities.forEach(e => {
                        try { cesiumViewer.entities.remove(e); } catch(ex) {}
                    });
                    cesiumEntities = [];
                }
                
                // Add hotspots based on current mode
                const hotspotsToAdd = data.hotspots;
                
                if (is3DMode && cesiumViewer) {
                    // 3D MODE - Add to Cesium globe
                    hotspotsToAdd.forEach((spot, i) => {
                        setTimeout(() => {
                            const entity = cesiumViewer.entities.add({
                                position: Cesium.Cartesian3.fromDegrees(spot.lng, spot.lat),
                                point: {
                                    pixelSize: 20,
                                    color: Cesium.Color.BLACK.withAlpha(0.8),
                                    outlineColor: Cesium.Color.DARKRED,
                                    outlineWidth: 3
                                },
                                label: {
                                    text: spot.region,
                                    font: '10px sans-serif',
                                    fillColor: Cesium.Color.WHITE,
                                    style: Cesium.LabelStyle.FILL,
                                    verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
                                    pixelOffset: new Cesium.Cartesian2(0, -25),
                                    show: false
                                }
                            });
                            cesiumEntities.push(entity);
                        }, i * 15);
                    });
                } else {
                    // 2D MODE - Add to Leaflet map with CircleMarkers for reliable positioning
                    log('Adding ' + hotspotsToAdd.length + ' hotspots to 2D map...', 'info');
                    
                    hotspotsToAdd.forEach((spot, i) => {
                        // Create CircleMarker with explicit lat/lng from API data
                        const lat = parseFloat(spot.lat);
                        const lng = parseFloat(spot.lng);
                        
                        // Validate coordinates
                        if (isNaN(lat) || isNaN(lng) || lat < -90 || lat > 90 || lng < -180 || lng > 180) {
                            console.warn('Invalid coordinates:', spot);
                            return;
                        }
                        
                        setTimeout(() => {
                            // Use CircleMarker for reliable global positioning
                            const circleMarker = L.circleMarker([lat, lng], {
                                radius: 8,
                                fillColor: '#ff3300',
                                color: '#000',
                                weight: 2,
                                opacity: 1,
                                fillOpacity: 0.8,
                                className: 'hotspot-circle'
                            }).bindPopup(
                                '<div style="min-width:150px;">' +
                                '<b style="color:#ff4444;font-size:14px;">🔥 ' + spot.region + '</b><br>' +
                                '<hr style="margin:5px 0;border-color:#333;">' +
                                '<b>Lat:</b> ' + lat.toFixed(4) + '<br>' +
                                '<b>Lng:</b> ' + lng.toFixed(4) + '<br>' +
                                '<b>Brightness:</b> ' + spot.brightness + 'K<br>' +
                                '<b>FRP:</b> ' + spot.frp + ' MW<br>' +
                                '<b>Satellite:</b> ' + spot.satellite + '<br>' +
                                '<b>Confidence:</b> ' + spot.confidence + '%' +
                                '</div>'
                            );
                            
                            circleMarker.addTo(map);
                            hotspotMarkers.push(circleMarker);
                            
                            // Log first few for debugging
                            if (i < 3) {
                                console.log('Added hotspot:', lat, lng, spot.region);
                            }
                        }, i * 5);
                    });
                    
                    // Zoom out to show global view after adding all markers
                    setTimeout(() => {
                        map.setView([20, 0], 2);
                        log('Map zoomed to global view', 'info');
                    }, hotspotsToAdd.length * 5 + 200);
                }
                
                log('NASA FIRMS: ' + data.total_count + ' hotspots detected globally', 'success');
                speak(data.total_count + ' wildfire hotspots detected worldwide.');
                
            } catch (e) {
                log('Hotspot fetch error: ' + e.message, 'error');
            }
        }
        
        // =============================================
        // DRONE DEPLOYMENT
        // =============================================
        
        async function deployDrones() {
            if (!selectedFireLocation) { log('Select a fire location first!', 'error'); return; }
            
            log('Deploying autonomous drone swarm...', 'info');
            speak('Deploying twelve drone units for fire suppression.');
            
            try {
                const response = await fetch('/api/deploy-drones', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ lat: selectedFireLocation.lat, lng: selectedFireLocation.lng, count: 12 })
                });
                
                const data = await response.json();
                data.drones.forEach((drone, i) => setTimeout(() => animateDrone(drone), i * 200));
                log('Mission ' + data.mission_id + ': ' + data.total_deployed + ' drones deployed', 'success');
                speak('Drone swarm deployed successfully.');
            } catch (e) { log('Drone deployment failed: ' + e.message, 'error'); }
        }
        
        function animateDrone(drone) {
            if (is3DMode && cesiumViewer) {
                // 3D drone animation
                const positions = drone.path.map(p => Cesium.Cartesian3.fromDegrees(p[1], p[0], 500));
                cesiumViewer.entities.add({
                    polyline: {
                        positions: positions,
                        width: 3,
                        material: new Cesium.PolylineGlowMaterialProperty({ glowPower: 0.3, color: Cesium.Color.CYAN })
                    }
                });
            } else {
                const path = drone.path;
                let step = 0;
                const icon = L.divIcon({ className: 'drone-marker', iconSize: [14, 14] });
                const marker = L.marker(path[0], { icon }).addTo(map);
                droneMarkers.push(marker);
                const trail = L.polyline([], { color: '#00f3ff', weight: 2, opacity: 0.6 }).addTo(map);
                droneMarkers.push(trail);
                const interval = setInterval(() => {
                    if (step >= path.length) { clearInterval(interval); return; }
                    marker.setLatLng(path[step]);
                    trail.addLatLng(path[step]);
                    step++;
                }, 100);
            }
        }
        
        // =============================================
        // CLOUD SEEDING
        // =============================================
        
        async function initiateCloudSeeding() {
            if (!selectedFireLocation) { log('Select a fire location first!', 'error'); return; }
            
            log('Initiating cloud seeding operation...', 'info');
            speak('Initiating cloud seeding operation.');
            
            try {
                const response = await fetch('/api/cloud-seeding', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ lat: selectedFireLocation.lat, lng: selectedFireLocation.lng })
                });
                
                const data = await response.json();
                
                data.seeding_points.forEach(point => {
                    if (is3DMode && cesiumViewer) {
                        cesiumViewer.entities.add({
                            position: Cesium.Cartesian3.fromDegrees(point.lng, point.lat, point.altitude_m),
                            point: { pixelSize: 20, color: Cesium.Color.PURPLE.withAlpha(0.7) },
                            label: { text: point.id, font: '12px sans-serif', fillColor: Cesium.Color.WHITE }
                        });
                    } else {
                        const marker = L.circleMarker([point.lat, point.lng], {
                            radius: 15, color: '#a855f7', fillColor: '#a855f7', fillOpacity: 0.4, weight: 2
                        }).addTo(map);
                        marker.bindPopup('<b style="color:#a855f7;">☁️ ' + point.id + '</b><br>Altitude: ' + point.altitude_m + 'm');
                        predictionPolygons.push(marker);
                    }
                });
                
                log('Operation ' + data.operation_id + ': Expected ' + data.expected_precipitation_mm + 'mm rain', 'success');
                speak('Cloud seeding initiated. Expecting ' + data.expected_precipitation_mm + ' millimeters of precipitation.');
            } catch (e) { log('Cloud seeding failed: ' + e.message, 'error'); }
        }
        
        // =============================================
        // LIVE WILDFIRE
        // =============================================
        
        async function goToLiveWildfire() {
            log('Fetching live NASA FIRMS data...', 'info');
            speak('Scanning for active wildfires.');
            
            try {
                const response = await fetch('/api/live-nasa-fires');
                const data = await response.json();
                activeFires = data.fires || [];
                
                fireMarkers.forEach(m => { if (map.hasLayer(m)) map.removeLayer(m); });
                fireMarkers = [];
                
                if (is3DMode && cesiumViewer) {
                    cesiumViewer.entities.removeAll();
                }
                
                activeFires.forEach(fire => {
                    if (is3DMode && cesiumViewer) {
                        cesiumViewer.entities.add({
                            position: Cesium.Cartesian3.fromDegrees(fire.longitude, fire.latitude),
                            point: { pixelSize: 18, color: Cesium.Color.ORANGE, outlineColor: Cesium.Color.RED, outlineWidth: 3 },
                            label: { text: fire.name, font: '12px sans-serif', fillColor: Cesium.Color.WHITE, verticalOrigin: Cesium.VerticalOrigin.BOTTOM, pixelOffset: new Cesium.Cartesian2(0, -20) }
                        });
                    } else {
                        const icon = L.divIcon({ className: 'fire-marker', iconSize: [20, 20] });
                        const marker = L.marker([fire.latitude, fire.longitude], { icon })
                            .bindPopup('<b style="color:#ff6600;">' + fire.name + '</b><br>Risk: ' + fire.risk_score + '%<br>Status: ' + fire.status)
                            .addTo(map);
                        marker.on('click', () => { selectedFireLocation = { lat: fire.latitude, lng: fire.longitude }; });
                        fireMarkers.push(marker);
                    }
                });
                
                if (activeFires.length > 0) {
                    const topFire = activeFires.reduce((max, f) => f.risk_score > max.risk_score ? f : max, activeFires[0]);
                    
                    if (is3DMode && cesiumViewer) {
                        cesiumViewer.camera.flyTo({
                            destination: Cesium.Cartesian3.fromDegrees(topFire.longitude, topFire.latitude, 1000000),
                            duration: 1
                        });
                    } else {
                        map.flyTo([topFire.latitude, topFire.longitude], 8, { duration: 1.5 });
                    }
                    
                    selectedFireLocation = { lat: topFire.latitude, lng: topFire.longitude };
                    log('Flying to: ' + topFire.name + ' (Risk: ' + topFire.risk_score + '%)', 'warning');
                    speak('Navigating to ' + topFire.name + '. Risk level is ' + topFire.risk_score + ' percent.');
                }
                
                log('NASA FIRMS: ' + activeFires.length + ' active fires detected', 'success');
            } catch (e) { log('Failed to fetch live fires: ' + e.message, 'error'); }
        }
        
        // =============================================
        // CITY SEARCH
        // =============================================
        
        async function searchCity() {
            const city = document.getElementById('citySearch').value.trim();
            if (!city) return;
            
            log('Searching for: ' + city + '...', 'info');
            
            try {
                const response = await fetch('/api/geocode?city=' + encodeURIComponent(city));
                const data = await response.json();
                
                if (data.found !== false) {
                    if (is3DMode && cesiumViewer) {
                        cesiumViewer.camera.flyTo({
                            destination: Cesium.Cartesian3.fromDegrees(data.lng, data.lat, 500000),
                            duration: 1
                        });
                    } else {
                        map.flyTo([data.lat, data.lng], 10, { duration: 1.5 });
                    }
                    selectedFireLocation = { lat: data.lat, lng: data.lng };
                    log('Navigated to: ' + data.name, 'success');
                    speak('Navigating to ' + city);
                } else {
                    log('Location not found: ' + city, 'warning');
                }
            } catch (e) { log('Search failed: ' + e.message, 'error'); }
        }
        
        // =============================================
        // EMBER SIMULATION
        // =============================================
        
        async function runEmberSimulation() {
            if (!selectedFireLocation) { log('Select a fire location first!', 'error'); return; }
            
            log('Running Ember-Cast trajectory simulation...', 'info');
            speak('Running ember trajectory simulation.');
            
            const windDirection = Math.random() * 360;
            const windRad = windDirection * Math.PI / 180;
            
            for (let i = 0; i < 8; i++) {
                const distance = 0.5 + Math.random() * 1.5;
                const spread = (Math.random() - 0.5) * 0.3;
                const endLat = selectedFireLocation.lat + (distance / 111) * Math.cos(windRad + spread);
                const endLng = selectedFireLocation.lng + (distance / 111) * Math.sin(windRad + spread);
                
                if (is3DMode && cesiumViewer) {
                    cesiumViewer.entities.add({
                        polyline: {
                            positions: [
                                Cesium.Cartesian3.fromDegrees(selectedFireLocation.lng, selectedFireLocation.lat, 100),
                                Cesium.Cartesian3.fromDegrees(endLng, endLat, 50)
                            ],
                            width: 2,
                            material: new Cesium.PolylineDashMaterialProperty({ color: Cesium.Color.ORANGE })
                        }
                    });
                    cesiumViewer.entities.add({
                        position: Cesium.Cartesian3.fromDegrees(endLng, endLat),
                        point: { pixelSize: 12, color: Cesium.Color.ORANGE }
                    });
                } else {
                    const path = [];
                    for (let t = 0; t <= 10; t++) {
                        const progress = t / 10;
                        path.push([
                            selectedFireLocation.lat + (endLat - selectedFireLocation.lat) * progress,
                            selectedFireLocation.lng + (endLng - selectedFireLocation.lng) * progress
                        ]);
                    }
                    const emberPath = L.polyline(path, { color: '#ff9500', weight: 2, opacity: 0.7, dashArray: '5, 5' }).addTo(map);
                    const spotMarker = L.circleMarker([endLat, endLng], { radius: 8, color: '#ff6600', fillColor: '#ff6600', fillOpacity: 0.5 }).addTo(map);
                    spotMarker.bindPopup('⚡ Potential Spot Fire<br>Distance: ' + distance.toFixed(1) + ' km');
                    predictionPolygons.push(emberPath, spotMarker);
                }
            }
            
            log('Ember simulation complete: 8 potential spot fires identified', 'success');
            speak('Eight potential spot fire locations identified.');
        }
        
        // =============================================
        // FUTURE PRESENCE MODE
        // =============================================
        
        async function enterFuturePresence() {
            if (!selectedFireLocation) {
                const center = map.getCenter();
                selectedFireLocation = { lat: center.lat, lng: center.lng };
            }
            
            log('Entering Future Presence Mode...', 'info');
            
            try {
                const response = await fetch('/api/future-presence', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ lat: selectedFireLocation.lat, lng: selectedFireLocation.lng, hours: 24 })
                });
                
                const data = await response.json();
                
                // Update overlay
                document.getElementById('futureTime').innerHTML = data.future_time + ' <span class="pulse"></span>';
                
                const forecastsContainer = document.getElementById('futureForecastsContainer');
                forecastsContainer.innerHTML = '';
                
                data.sensory_forecasts.forEach((forecast, i) => {
                    const div = document.createElement('div');
                    div.className = 'future-forecast-item';
                    div.style.animationDelay = (0.5 + i * 0.7) + 's';
                    div.textContent = forecast;
                    forecastsContainer.appendChild(div);
                });
                
                // Show overlay
                document.getElementById('futureOverlay').classList.add('visible');
                
                speak('Entering future presence mode. ' + data.sensory_forecasts[0]);
                
            } catch (e) {
                log('Future Presence error: ' + e.message, 'error');
            }
        }
        
        function closeFuturePresence() {
            document.getElementById('futureOverlay').classList.remove('visible');
            log('Returned to present', 'info');
        }
        
        // =============================================
        // WILDFIRE NEWS TRENDS
        // =============================================
        
        async function showWildfireNewsTrends() {
            log('Loading global wildfire news trends...', 'info');
            speak('Opening wildfire global trends dashboard with live news reports.');
            
            try {
                const response = await fetch('/api/wildfire-news-trends');
                const data = await response.json();
                
                // Populate statistics
                const statsHtml = `
                    <div class="stat-card">
                        <div class="value">${data.statistics.active_fires_worldwide}</div>
                        <div class="label">Active Fires Worldwide</div>
                    </div>
                    <div class="stat-card">
                        <div class="value">${data.statistics.fires_contained_today}</div>
                        <div class="label">Contained Today</div>
                    </div>
                    <div class="stat-card">
                        <div class="value">${data.statistics.new_fires_reported}</div>
                        <div class="label">New Fires Reported</div>
                    </div>
                    <div class="stat-card">
                        <div class="value">${data.statistics.total_area_burning}</div>
                        <div class="label">Total Area Burning</div>
                    </div>
                    <div class="stat-card">
                        <div class="value">${data.statistics.air_quality_alerts}</div>
                        <div class="label">Air Quality Alerts</div>
                    </div>
                    <div class="stat-card">
                        <div class="value">${data.statistics.evacuation_orders}</div>
                        <div class="label">Evacuation Orders</div>
                    </div>
                `;
                document.getElementById('newsStats').innerHTML = statsHtml;
                
                // Populate news feed
                let newsHtml = '';
                data.news.forEach(item => {
                    const categoryClass = item.category.toLowerCase().replace(' ', '-');
                    const cardClass = categoryClass;
                    newsHtml += `
                        <div class="news-card ${cardClass}">
                            <div class="news-card-header">
                                <span class="news-category ${categoryClass}">${item.category}</span>
                                <span class="news-source">${item.source} · ${item.time}</span>
                            </div>
                            <h3>${item.title}</h3>
                            <p>${item.summary}</p>
                            <div class="news-meta">
                                <div class="news-meta-item">
                                    <span>Location</span>
                                    <strong>${item.location}</strong>
                                </div>
                                <div class="news-meta-item">
                                    <span>Area Affected</span>
                                    <strong>${item.affected_area}</strong>
                                </div>
                                <div class="news-meta-item">
                                    <span>Containment</span>
                                    <strong>${item.containment}</strong>
                                </div>
                            </div>
                        </div>
                    `;
                });
                document.getElementById('newsFeedContainer').innerHTML = newsHtml;
                
                // Populate regional hotspots
                let hotspotsHtml = '';
                data.regional_hotspots.forEach(region => {
                    const severityClass = region.severity.toLowerCase();
                    const trendIcon = region.trend === 'increasing' ? '↑' : region.trend === 'decreasing' ? '↓' : '→';
                    hotspotsHtml += `
                        <div class="hotspot-card">
                            <div>
                                <div class="region">${region.region}</div>
                                <div class="fires">${region.active_fires} active fires</div>
                            </div>
                            <div style="text-align: right;">
                                <span class="severity-badge ${severityClass}">${region.severity}</span>
                                <div class="trend-indicator ${region.trend}">${trendIcon} ${region.trend}</div>
                            </div>
                        </div>
                    `;
                });
                document.getElementById('regionalHotspotsContainer').innerHTML = hotspotsHtml;
                
                // Show overlay
                document.getElementById('newsOverlay').classList.add('visible');
                log('News dashboard loaded with ' + data.news.length + ' reports', 'success');
                
            } catch (e) {
                log('News trends error: ' + e.message, 'error');
            }
        }
        
        function closeNewsTrends() {
            document.getElementById('newsOverlay').classList.remove('visible');
            log('Closed news dashboard', 'info');
        }
        
        // =============================================
        // 3D GLOBE - FAST LOADING
        // =============================================
        
        async function toggle3DGlobe() {
            is3DMode = !is3DMode;
            const btn = document.getElementById('btn3D');
            
            if (is3DMode) {
                document.getElementById('leaflet-map').style.display = 'none';
                document.getElementById('cesium-globe').style.display = 'block';
                document.getElementById('loading3D').classList.add('visible');
                btn.classList.add('active');
                
                // Wait for Cesium to load if not already
                await preloadCesium();
                
                if (!cesiumViewer) {
                    // Create viewer with minimal options for fast loading
                    cesiumViewer = new Cesium.Viewer('cesium-globe', {
                        imageryProvider: new Cesium.ArcGisMapServerImageryProvider({
                            url: 'https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer'
                        }),
                        baseLayerPicker: false,
                        geocoder: false,
                        homeButton: false,
                        sceneModePicker: false,
                        navigationHelpButton: false,
                        animation: false,
                        timeline: false,
                        fullscreenButton: false,
                        vrButton: false,
                        infoBox: false,
                        selectionIndicator: false,
                        creditContainer: document.createElement('div'),
                        requestRenderMode: true,  // Only render when needed
                        maximumRenderTimeChange: Infinity
                    });
                    
                    cesiumViewer.scene.globe.enableLighting = false;
                    cesiumViewer.scene.fog.enabled = false;
                    cesiumViewer.scene.globe.showGroundAtmosphere = false;
                    
                    // Mouse move for coords
                    cesiumViewer.screenSpaceEventHandler.setInputAction(function(movement) {
                        const cartesian = cesiumViewer.camera.pickEllipsoid(movement.endPosition);
                        if (cartesian) {
                            const carto = Cesium.Cartographic.fromCartesian(cartesian);
                            document.getElementById('coords').textContent = 
                                'LAT: ' + Cesium.Math.toDegrees(carto.latitude).toFixed(4) + 
                                ' | LNG: ' + Cesium.Math.toDegrees(carto.longitude).toFixed(4);
                        }
                    }, Cesium.ScreenSpaceEventType.MOUSE_MOVE);
                    
                    // Click for selection
                    cesiumViewer.screenSpaceEventHandler.setInputAction(function(click) {
                        const cartesian = cesiumViewer.camera.pickEllipsoid(click.position);
                        if (cartesian) {
                            const carto = Cesium.Cartographic.fromCartesian(cartesian);
                            selectedFireLocation = {
                                lat: Cesium.Math.toDegrees(carto.latitude),
                                lng: Cesium.Math.toDegrees(carto.longitude)
                            };
                            log('Location selected: ' + selectedFireLocation.lat.toFixed(4) + ', ' + selectedFireLocation.lng.toFixed(4), 'info');
                        }
                    }, Cesium.ScreenSpaceEventType.LEFT_CLICK);
                }
                
                // Fly to current map position
                const center = map.getCenter();
                cesiumViewer.camera.flyTo({
                    destination: Cesium.Cartesian3.fromDegrees(center.lng, center.lat, 5000000),
                    duration: 0.5  // Fast transition
                });
                
                document.getElementById('loading3D').classList.remove('visible');
                log('3D Holographic Mode ACTIVATED', 'success');
                speak('3D holographic mode activated.');
            } else {
                document.getElementById('cesium-globe').style.display = 'none';
                document.getElementById('leaflet-map').style.display = 'block';
                btn.classList.remove('active');
                log('2D Map Mode', 'info');
            }
        }
        
        function toggleSatellite() {
            if (isSatellite) {
                map.removeLayer(satelliteLayer);
                map.addLayer(streetLayer);
                log('Street map view');
            } else {
                map.removeLayer(streetLayer);
                map.addLayer(satelliteLayer);
                log('Satellite view');
            }
            isSatellite = !isSatellite;
        }
        
        // =============================================
        // UI HELPERS
        // =============================================
        
        function toggleExpand(sectionId) {
            document.getElementById(sectionId).classList.toggle('open');
        }
        
        function log(msg, type = '') {
            const terminal = document.getElementById('terminal');
            const div = document.createElement('div');
            div.className = type;
            div.textContent = '> ' + msg;
            terminal.appendChild(div);
            terminal.scrollTop = terminal.scrollHeight;
        }
        
        map.on('mousemove', (e) => {
            document.getElementById('coords').textContent = 'LAT: ' + e.latlng.lat.toFixed(4) + ' | LNG: ' + e.latlng.lng.toFixed(4);
        });
        
        map.on('click', (e) => {
            selectedFireLocation = { lat: e.latlng.lat, lng: e.latlng.lng };
        });
        
        // Initialize
        log('WildFire AI v6.0 - Future Presence Mode Ready', 'success');
    </script>
</body>
</html>
"""

# ==========================================
# RUN SERVER
# ==========================================

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("🔥 WILDFIRE AI - FUTURE PRESENCE MODE v6.0")
    print("="*60)
    print("\n📡 Server starting at: http://localhost:8000")
    print("\nFeatures:")
    print("  ✦ Physics-Informed Fire Spread Prediction")
    print("  ✦ Future Presence Mode - Experiential Forecasting")
    print("  ✦ NASA FIRMS Global Hotspots (Dark Blooming)")
    print("  ✦ Autonomous Drone Swarm Deployment")
    print("  ✦ Cloud Seeding Operations")
    print("  ✦ Fast 3D Holographic Globe (~1 sec load)")
    print("  ✦ All features work in both 2D and 3D modes")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
