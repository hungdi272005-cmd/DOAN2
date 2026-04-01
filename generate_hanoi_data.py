import pandas as pd
import numpy as np
from datetime import timedelta, datetime

# Parameters
np.random.seed(42)
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 12, 31, 23, 0)
routes = ['Nguyen Trai', 'Vanh Dai 3', 'Ton Duc Thang']
weather_conditions = ['Clear', 'Rain', 'Heavy Rain']
weather_probs = [0.8, 0.15, 0.05]

# Generate Hourly timestamps
timestamps = pd.date_range(start=start_date, end=end_date, freq='H')

data = []

for route in routes:
    for ts in timestamps:
        hour = ts.hour
        day_of_week = ts.dayofweek # 0=Mon, 6=Sun
        is_weekend = day_of_week >= 5
        
        # Base speed depending on route
        if route == 'Nguyen Trai':
            base_speed = 40
        elif route == 'Vanh Dai 3':
            base_speed = 60
        else: # Ton Duc Thang
            base_speed = 35
            
        # Determine rush hour impact
        rush_hour_penalty = 1.0
        if not is_weekend:
            if 7 <= hour <= 9: # Morning rush
                rush_hour_penalty = np.random.uniform(0.15, 0.3) # Drop to 15-30% of base speed
            elif 17 <= hour <= 19: # Evening rush
                rush_hour_penalty = np.random.uniform(0.1, 0.25) # Drop to 10-25% of base speed
            elif 9 < hour < 17: # Daytime
                rush_hour_penalty = np.random.uniform(0.5, 0.7)
            else: # Night time
                rush_hour_penalty = np.random.uniform(0.9, 1.1)
        else:
            # Weekend is generally smoother but slower mid-day
            if 10 <= hour <= 20:
                rush_hour_penalty = np.random.uniform(0.6, 0.8)
            else:
                rush_hour_penalty = np.random.uniform(0.9, 1.1)
                
        # Determine weather condition
        weather = np.random.choice(weather_conditions, p=weather_probs)
        weather_penalty = 1.0
        if weather == 'Rain':
            weather_penalty = np.random.uniform(0.6, 0.8)
        elif weather == 'Heavy Rain':
            weather_penalty = np.random.uniform(0.3, 0.5)
            
        # Calculate final speed with some random noise
        noise = np.random.uniform(-0.05, 0.05)
        final_speed = base_speed * rush_hour_penalty * weather_penalty * (1 + noise)
        
        # Ensure minimum speed of 3 km/h (heavy congestion)
        final_speed = max(3.0, final_speed)
        
        # Calculate delay per km in minutes (if ideal speed is base_speed)
        ideal_time_min_per_km = 60 / base_speed
        actual_time_min_per_km = 60 / final_speed
        delay_per_km = actual_time_min_per_km - ideal_time_min_per_km
        delay_per_km = max(0.0, delay_per_km)
        
        data.append({
            'timestamp': ts,
            'route': route,
            'speed_kmh': round(final_speed, 2),
            'delay_per_km_min': round(delay_per_km, 2),
            'weather': weather,
            'is_weekend': int(is_weekend)
        })

df = pd.DataFrame(data)
df.to_csv('trafficstats_hanoi_mock.csv', index=False)
print(f"Generated data for {len(routes)} routes over {len(timestamps)} hours.")
print("Total rows:", len(df))
print("File saved as 'trafficstats_hanoi_mock.csv'")
