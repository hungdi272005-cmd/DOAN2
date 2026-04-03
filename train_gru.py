"""
train_gru.py - Huấn luyện model GRU Deep Learning
Hỗ trợ cả dữ liệu mock và dữ liệu thật từ TomTom

Sử dụng:
  python train_gru.py              # Dùng mock data (mặc định)
  python train_gru.py --realtime   # Dùng dữ liệu thật TomTom
  python train_gru.py --combined   # Kết hợp cả mock + thật
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
import joblib
import argparse
import os

def load_data(data_source='mock'):
    """Load dữ liệu theo nguồn."""
    if data_source == 'realtime':
        filepath = 'hanoi_traffic_realtime.csv'
        if not os.path.exists(filepath):
            print("Chua co file du lieu that. Chay 'python tomtom_collector.py' truoc!")
            return None
        print(f"Dang dung du lieu THAT tu TomTom: {filepath}")
        df = pd.read_csv(filepath)
    elif data_source == 'combined':
        dfs = []
        if os.path.exists('trafficstats_hanoi_mock.csv'):
            df_mock = pd.read_csv('trafficstats_hanoi_mock.csv')
            df_mock['data_source'] = 'mock'
            dfs.append(df_mock)
            print(f"   Mock data: {len(df_mock)} records")
        if os.path.exists('hanoi_traffic_realtime.csv'):
            df_real = pd.read_csv('hanoi_traffic_realtime.csv')
            dfs.append(df_real)
            print(f"   Real data: {len(df_real)} records")
        if not dfs:
            print("Khong tim thay file du lieu nao!")
            return None
        df = pd.concat(dfs, ignore_index=True)
        print(f"Ket hop: Tong {len(df)} records")
    else:
        filepath = 'trafficstats_hanoi_mock.csv'
        print(f"Dang dung du lieu MOCK: {filepath}")
        df = pd.read_csv(filepath)
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def train_gru_model(df):
    """Huấn luyện GRU model."""
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

    features = ['route_encoded', 'weather_encoded', 'is_weekend', 'hour', 'day_of_week', 'month']
    X = df[features].values
    y = df['speed_kmh'].values

    # Standardize features
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    joblib.dump(scaler_X, 'scaler_X_gru.pkl')

    # Reshape input to be 3D [samples, timesteps, features]
    X_gru = np.reshape(X_scaled, (X_scaled.shape[0], 1, X_scaled.shape[1]))

    # Split chronological
    split_idx = int(len(X_gru) * 0.8)
    X_train, X_test = X_gru[:split_idx], X_gru[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"   Train: {X_train.shape}")
    print(f"   Test: {X_test.shape}")

    print("Building GRU model...")
    gru_model = Sequential([
        GRU(128, input_shape=(X_train.shape[1], X_train.shape[2]), activation='tanh', return_sequences=True),
        Dropout(0.3),
        GRU(64, activation='tanh'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    gru_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

    print("Training GRU... (50 epochs, auto-stop khi hoi tu)")
    gru_model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=64,
        validation_data=(X_test, y_test),
        callbacks=[early_stop, reduce_lr]
    )

    print("Evaluating...")
    loss, mae = gru_model.evaluate(X_test, y_test)
    print(f"Test RMSE: {np.sqrt(loss):.2f}")
    print(f"Test MAE: {mae:.2f}")

    print("Saving GRU model...")
    gru_model.save('gru_traffic_model.keras')
    print("Xong! Model GRU da duoc huan luyen thanh cong.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GRU model')
    parser.add_argument('--realtime', action='store_true', help='Dung du lieu that tu TomTom')
    parser.add_argument('--combined', action='store_true', help='Ket hop mock + that')
    args = parser.parse_args()

    if args.realtime:
        data_source = 'realtime'
    elif args.combined:
        data_source = 'combined'
    else:
        data_source = 'mock'

    print("=" * 60)
    print("  TRAIN GRU - DEEP LEARNING")
    print("=" * 60)
    
    df = load_data(data_source)
    if df is not None:
        train_gru_model(df)
