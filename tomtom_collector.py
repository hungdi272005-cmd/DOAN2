"""
tomtom_collector.py - Thu thập dữ liệu giao thông THẬT từ TomTom API + Thời tiết từ OpenWeatherMap
Sử dụng:
  python tomtom_collector.py              # Thu thập 1 lần (test)
  python tomtom_collector.py --schedule   # Chạy liên tục mỗi 15 phút
  python tomtom_collector.py --backfill   # Thu thập liên tục 24h để tạo dataset ban đầu
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime
import time
import os
import sys
import argparse
from dotenv import load_dotenv

# Fix Windows console encoding
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Load API keys từ .env
load_dotenv()
TOMTOM_API_KEY = os.getenv('TOMTOM_API_KEY')
OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY')

# === CẤU HÌNH TUYẾN ĐƯỜNG HÀ NỘI ===
HANOI_ROUTES = {
    'Nguyen Trai': {'lat': 21.0024, 'lon': 105.7979, 'description': 'Nguyễn Trãi - Thanh Xuân'},
    'Vanh Dai 3': {'lat': 20.9952, 'lon': 105.7872, 'description': 'Vành Đai 3 - Linh Đàm'},
    'Ton Duc Thang': {'lat': 21.0256, 'lon': 105.8365, 'description': 'Tôn Đức Thắng - Đống Đa'},
}

# Bounding box Hà Nội cho Incidents API (minLat, minLon, maxLat, maxLon)
HANOI_BBOX = '20.95,105.75,21.08,105.90'

# Tọa độ trung tâm Hà Nội cho Weather API
HANOI_CENTER = {'lat': 21.0285, 'lon': 105.8542}

OUTPUT_FILE = 'hanoi_traffic_realtime.csv'


def get_weather():
    """Lấy thời tiết hiện tại ở Hà Nội từ OpenWeatherMap API."""
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather"
        params = {
            'lat': HANOI_CENTER['lat'],
            'lon': HANOI_CENTER['lon'],
            'appid': OPENWEATHER_API_KEY,
            'units': 'metric',
            'lang': 'vi'
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Phân loại thời tiết
        weather_id = data['weather'][0]['id']
        weather_main = data['weather'][0]['main']
        temp = data['main']['temp']
        humidity = data['main']['humidity']
        wind_speed = data['wind']['speed']
        weather_desc = data['weather'][0]['description']

        # Map sang 3 loại: Clear, Rain, Heavy Rain
        if weather_id >= 500 and weather_id < 510:
            if weather_id >= 502:  # Heavy rain
                weather_category = 'Heavy Rain'
            else:
                weather_category = 'Rain'
        elif weather_id >= 200 and weather_id < 300:  # Thunderstorm
            weather_category = 'Heavy Rain'
        elif weather_id >= 300 and weather_id < 400:  # Drizzle
            weather_category = 'Rain'
        elif weather_id >= 511 and weather_id < 600:  # Snow/Freezing rain
            weather_category = 'Heavy Rain'
        else:
            weather_category = 'Clear'

        print(f"  🌤️  Thời tiết: {weather_desc} ({weather_category}), {temp}°C, Gió: {wind_speed}m/s")
        return {
            'weather': weather_category,
            'weather_detail': weather_desc,
            'temperature': temp,
            'humidity': humidity,
            'wind_speed': wind_speed
        }
    except Exception as e:
        print(f"  ⚠️  Lỗi lấy thời tiết: {e}")
        return {
            'weather': 'Clear',
            'weather_detail': 'unknown',
            'temperature': None,
            'humidity': None,
            'wind_speed': None
        }


def get_traffic_flow(route_name, lat, lon):
    """Lấy dữ liệu Traffic Flow cho 1 tuyến đường từ TomTom API."""
    try:
        url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
        params = {
            'key': TOMTOM_API_KEY,
            'point': f'{lat},{lon}',
            'unit': 'KMPH'
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        flow = data['flowSegmentData']
        current_speed = flow['currentSpeed']
        free_flow_speed = flow['freeFlowSpeed']
        current_travel_time = flow['currentTravelTime']
        free_flow_travel_time = flow['freeFlowTravelTime']
        confidence = flow['confidence']
        road_closure = flow.get('roadClosure', False)
        frc = flow.get('frc', 'unknown')

        # Tính congestion ratio
        congestion_ratio = current_speed / free_flow_speed if free_flow_speed > 0 else 1.0

        print(f"  🚗 {route_name}: {current_speed} km/h (tự do: {free_flow_speed} km/h) | "
              f"Ratio: {congestion_ratio:.2f} | Confidence: {confidence:.2f}")

        return {
            'current_speed': current_speed,
            'free_flow_speed': free_flow_speed,
            'current_travel_time': current_travel_time,
            'free_flow_travel_time': free_flow_travel_time,
            'confidence': confidence,
            'road_closure': road_closure,
            'frc': frc,
            'congestion_ratio': congestion_ratio
        }
    except Exception as e:
        print(f"  ❌ Lỗi lấy traffic flow cho {route_name}: {e}")
        return None


def get_incidents():
    """Lấy số lượng sự cố giao thông trong khu vực Hà Nội."""
    try:
        # TomTom Incidents v5 - dùng POST với bbox
        url = f"https://api.tomtom.com/traffic/services/5/incidentDetails"
        params = {
            'key': TOMTOM_API_KEY,
            'bbox': HANOI_BBOX,
            'language': 'vi',
            'categoryFilter': '0,1,2,3,4,5,6,7,8,9,10,11,14',
            'timeValidityFilter': 'present'
        }
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        incidents = data.get('incidents', [])
        count = len(incidents)
        print(f"  🚨 Sự cố giao thông trong khu vực: {count}")
        return count
    except Exception as e:
        print(f"  ⚠️  Lỗi lấy incidents: {e} (bỏ qua)")
        return 0


def collect_once():
    """Thu thập dữ liệu 1 lần cho tất cả tuyến đường."""
    now = datetime.now()
    print(f"\n{'='*60}")
    print(f"⏰ Thu thập dữ liệu: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    # Lấy thời tiết chung
    weather_info = get_weather()

    # Lấy incidents chung
    incident_count = get_incidents()

    # Lấy traffic flow cho từng tuyến
    records = []
    for route_name, route_info in HANOI_ROUTES.items():
        flow_data = get_traffic_flow(route_name, route_info['lat'], route_info['lon'])

        if flow_data is None:
            continue

        # Tính speed_kmh (tương thích với model cũ) = current_speed
        is_weekend = 1 if now.weekday() >= 5 else 0

        # Tính delay_per_km
        if flow_data['current_speed'] > 0:
            actual_time = 60 / flow_data['current_speed']
        else:
            actual_time = 60 / 3  # minimum 3 km/h
        if flow_data['free_flow_speed'] > 0:
            ideal_time = 60 / flow_data['free_flow_speed']
        else:
            ideal_time = 1
        delay_per_km = max(0, actual_time - ideal_time)

        record = {
            'timestamp': now.strftime('%Y-%m-%d %H:%M:%S'),
            'route': route_name,
            'speed_kmh': round(flow_data['current_speed'], 2),
            'free_flow_speed': round(flow_data['free_flow_speed'], 2),
            'delay_per_km_min': round(delay_per_km, 2),
            'current_travel_time': flow_data['current_travel_time'],
            'free_flow_travel_time': flow_data['free_flow_travel_time'],
            'confidence': round(flow_data['confidence'], 2),
            'congestion_ratio': round(flow_data['congestion_ratio'], 4),
            'road_closure': int(flow_data['road_closure']),
            'incident_count': incident_count,
            'weather': weather_info['weather'],
            'weather_detail': weather_info['weather_detail'],
            'temperature': weather_info['temperature'],
            'humidity': weather_info['humidity'],
            'wind_speed': weather_info['wind_speed'],
            'is_weekend': is_weekend,
            'hour': now.hour,
            'day_of_week': now.weekday(),
            'month': now.month,
            'data_source': 'tomtom_realtime'
        }
        records.append(record)

    if not records:
        print("❌ Không thu thập được dữ liệu nào!")
        return 0

    # Ghi vào CSV (append)
    df_new = pd.DataFrame(records)
    file_exists = os.path.exists(OUTPUT_FILE)

    if file_exists:
        df_new.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
    else:
        df_new.to_csv(OUTPUT_FILE, mode='w', header=True, index=False)

    # Đếm tổng records
    if file_exists:
        total_lines = sum(1 for _ in open(OUTPUT_FILE)) - 1  # trừ header
    else:
        total_lines = len(records)

    print(f"\n✅ Đã lưu {len(records)} records → {OUTPUT_FILE} (Tổng: {total_lines} records)")
    return len(records)


def run_scheduled(interval_minutes=15):
    """Chạy thu thập dữ liệu theo lịch."""
    print(f"🔄 Bắt đầu thu thập tự động mỗi {interval_minutes} phút")
    print(f"   Nhấn Ctrl+C để dừng\n")

    while True:
        try:
            collect_once()
            print(f"\n⏳ Chờ {interval_minutes} phút đến lần thu thập tiếp...")
            time.sleep(interval_minutes * 60)
        except KeyboardInterrupt:
            print("\n\n🛑 Đã dừng thu thập dữ liệu.")
            break
        except Exception as e:
            print(f"\n⚠️ Lỗi: {e}")
            print(f"   Thử lại sau {interval_minutes} phút...")
            time.sleep(interval_minutes * 60)


def backfill_collect(hours=24, interval_minutes=15):
    """Thu thập liên tục trong N giờ để tạo dataset ban đầu."""
    total_iterations = (hours * 60) // interval_minutes
    print(f"📊 Backfill: Thu thập {total_iterations} lần trong {hours} giờ (mỗi {interval_minutes} phút)")

    for i in range(total_iterations):
        print(f"\n--- Lần {i+1}/{total_iterations} ---")
        collect_once()
        if i < total_iterations - 1:
            print(f"⏳ Chờ {interval_minutes} phút...")
            time.sleep(interval_minutes * 60)

    print(f"\n🎉 Hoàn thành backfill! Đã thu thập {total_iterations * len(HANOI_ROUTES)} records")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Thu thập dữ liệu giao thông Hà Nội từ TomTom API')
    parser.add_argument('--schedule', action='store_true', help='Chạy liên tục mỗi 15 phút')
    parser.add_argument('--backfill', action='store_true', help='Thu thập liên tục 24h')
    parser.add_argument('--interval', type=int, default=15, help='Khoảng cách giữa các lần thu thập (phút)')
    parser.add_argument('--hours', type=int, default=24, help='Số giờ backfill')
    args = parser.parse_args()

    if not TOMTOM_API_KEY:
        print("❌ Chưa cấu hình TOMTOM_API_KEY trong file .env!")
        sys.exit(1)
    if not OPENWEATHER_API_KEY:
        print("❌ Chưa cấu hình OPENWEATHER_API_KEY trong file .env!")
        sys.exit(1)

    print("🚦 TomTom Traffic Data Collector - Hà Nội")
    print(f"   API Key TomTom: {TOMTOM_API_KEY[:8]}...{TOMTOM_API_KEY[-4:]}")
    print(f"   API Key Weather: {OPENWEATHER_API_KEY[:8]}...{OPENWEATHER_API_KEY[-4:]}")
    print(f"   Tuyến đường: {', '.join(HANOI_ROUTES.keys())}")

    if args.schedule:
        run_scheduled(args.interval)
    elif args.backfill:
        backfill_collect(args.hours, args.interval)
    else:
        collect_once()
