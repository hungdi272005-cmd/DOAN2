"""
traffic_pipeline.py - Pipeline huấn luyện Random Forest
Hỗ trợ cả dữ liệu mock (cũ) và dữ liệu thật từ TomTom (mới)

Sử dụng:
  python traffic_pipeline.py              # Dùng mock data (mặc định)
  python traffic_pipeline.py --realtime   # Dùng dữ liệu thật TomTom
  python traffic_pipeline.py --combined   # Kết hợp cả mock + thật
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import argparse
import os

def load_data(data_source='realtime'):
    """Load dữ liệu theo nguồn và gộp dữ liệu TomTom Traffic Index."""
    from data_utils import load_and_merge_data
    try:
        # Tự động gộp dữ liệu
        df = load_and_merge_data(data_source)
        return df
    except Exception as e:
        print(f"❌ Lỗi tải dữ liệu: {e}")
        return None

def train_random_forest(df):
    """Huấn luyện model Random Forest."""
    print("\nFeature Engineering...")
    
    # Extract time features (nếu chưa có)
    if 'hour' not in df.columns:
        df['hour'] = df['timestamp'].dt.hour
    if 'day_of_week' not in df.columns:
        df['day_of_week'] = df['timestamp'].dt.dayofweek
    if 'month' not in df.columns:
        df['month'] = df['timestamp'].dt.month

    # Encode categorical features
    le_route = LabelEncoder()
    df['route_encoded'] = le_route.fit_transform(df['route'])
    joblib.dump(le_route, 'le_route.pkl')
    print(f"   Routes: {list(le_route.classes_)}")

    le_weather = LabelEncoder()
    df['weather_encoded'] = le_weather.fit_transform(df['weather'])
    joblib.dump(le_weather, 'le_weather.pkl')
    print(f"   Weather: {list(le_weather.classes_)}")

    # Define features and target
    features = [
        'route_encoded', 'weather_encoded', 'is_weekend', 'hour', 'day_of_week', 'month',
        'TrafficIndexLive', 'JamsCount', 'JamsLengthInKms', 'JamsDelay'
    ]
    target = 'speed_kmh'

    X = df[features]
    y = df[target]

    print(f"\nSplitting data... (Total: {len(X)} samples)")
    # Chronological split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"   Train shape: {X_train.shape}")
    print(f"   Test shape: {X_test.shape}")

    print("\nTraining Random Forest model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    print("\nEvaluating...")
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)

    print(f"   Test RMSE (km/h): {rmse:.2f}")
    print(f"   Test MAE (km/h): {mae:.2f}")

    print("\nSaving model...")
    joblib.dump(model, 'rf_traffic_model.pkl')
    
    # Lưu scaler cho Random Forest (dùng trong app)
    scaler_X = StandardScaler()
    scaler_X.fit(X_train)
    joblib.dump(scaler_X, 'scaler_X_rf.pkl')
    
    print("Pipeline completed successfully!")
    return model, rmse, mae

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Traffic Pipeline - Random Forest')
    parser.add_argument('--realtime', action='store_true', help='Dung du lieu that tu TomTom')
    parser.add_argument('--combined', action='store_true', help='Ket hop mock + that')
    args = parser.parse_args()

    if args.realtime or (not args.combined):
        data_source = 'realtime'
    else:
        data_source = 'combined'

    print("=" * 60)
    print("  TRAFFIC PIPELINE - RANDOM FOREST")
    print("=" * 60)
    
    df = load_data(data_source)
    if df is not None:
        train_random_forest(df)
        
        # Tự động đánh giá lại các mô hình sau khi huấn luyện xong
        print("\n" + "="*40)
        print("📊 Tự động chạy đánh giá độ chính xác (Accuracy, Precision, Recall, F1)...")
        try:
            from evaluate_models import evaluate_all_models
            evaluate_all_models(data_source)
        except Exception as e:
            print(f"⚠️ Không thể tự động chạy đánh giá: {e}")
