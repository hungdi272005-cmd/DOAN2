"""
train_lstm.py - Huấn luyện model LSTM Deep Learning
Hỗ trợ cả dữ liệu mock và dữ liệu thật từ TomTom

Sử dụng:
  python train_lstm.py              # Dùng mock data (mặc định)
  python train_lstm.py --realtime   # Dùng dữ liệu thật TomTom
  python train_lstm.py --combined   # Kết hợp cả mock + thật
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
import joblib
import argparse
import os
import json

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

def train_lstm_model(df):
    """Huấn luyện LSTM model."""
    print("Loading and preparing data...")
    
    # Build features
    if 'hour' not in df.columns:
        df['hour'] = df['timestamp'].dt.hour
    if 'day_of_week' not in df.columns:
        df['day_of_week'] = df['timestamp'].dt.dayofweek
    if 'month' not in df.columns:
        df['month'] = df['timestamp'].dt.month

    le_route = joblib.load('le_route.pkl')
    le_weather = joblib.load('le_weather.pkl')

    df['route_encoded'] = le_route.transform(df['route'])
    df['weather_encoded'] = le_weather.transform(df['weather'])

    features = [
        'route_encoded', 'weather_encoded', 'is_weekend', 'hour', 'day_of_week', 'month',
        'TrafficIndexLive', 'JamsCount', 'JamsLengthInKms', 'JamsDelay'
    ]
    X = df[features].values
    y = df['speed_kmh'].values

    # Standardize features
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    joblib.dump(scaler_X, 'scaler_X_lstm.pkl')

    # Reshape input to be 3D [samples, timesteps, features]
    X_lstm = np.reshape(X_scaled, (X_scaled.shape[0], 1, X_scaled.shape[1]))

    # Split chronological
    split_idx = int(len(X_lstm) * 0.8)
    X_train, X_test = X_lstm[:split_idx], X_lstm[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"   Train: {X_train.shape}")
    print(f"   Test: {X_test.shape}")

    print("Building LSTM model...")
    lstm_model = Sequential([
        LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), activation='tanh', return_sequences=True),
        Dropout(0.3),
        LSTM(64, activation='tanh'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    lstm_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

    print("Training LSTM... (50 epochs, auto-stop khi hoi tu)")
    history = lstm_model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=64,
        validation_data=(X_test, y_test),
        callbacks=[early_stop, reduce_lr]
    )

    # Lưu lịch sử huấn luyện để vẽ biểu đồ đường cong học tập
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    history_dict['epochs'] = list(range(1, len(history_dict['loss']) + 1))
    with open('lstm_training_history.json', 'w', encoding='utf-8') as f:
        json.dump(history_dict, f, ensure_ascii=False, indent=2)
    print(f"   Đã lưu lịch sử huấn luyện ({len(history_dict['epochs'])} epochs) vào lstm_training_history.json")

    print("Evaluating...")
    loss, mae = lstm_model.evaluate(X_test, y_test)
    print(f"Test RMSE: {np.sqrt(loss):.2f}")
    print(f"Test MAE: {mae:.2f}")

    print("Saving LSTM model...")
    lstm_model.save('lstm_traffic_model.keras')
    print("Xong! Model LSTM da duoc huan luyen thanh cong.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train LSTM model')
    parser.add_argument('--realtime', action='store_true', help='Dung du lieu that tu TomTom')
    parser.add_argument('--combined', action='store_true', help='Ket hop mock + that')
    args = parser.parse_args()

    if args.realtime or (not args.combined):
        data_source = 'realtime'
    else:
        data_source = 'combined'

    print("=" * 60)
    print("  TRAIN LSTM - DEEP LEARNING")
    print("=" * 60)
    
    df = load_data(data_source)
    if df is not None:
        train_lstm_model(df)
        
        # Tự động đánh giá lại các mô hình sau khi huấn luyện xong
        print("\n" + "="*40)
        print("📊 Tự động chạy đánh giá độ chính xác (Accuracy, Precision, Recall, F1)...")
        try:
            from evaluate_models import evaluate_all_models
            evaluate_all_models(data_source)
        except Exception as e:
            print(f"⚠️ Không thể tự động chạy đánh giá: {e}")
