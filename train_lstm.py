import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler
import joblib

print("Loading data...")
df = pd.read_csv('trafficstats_hanoi_mock.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Build features
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['month'] = df['timestamp'].dt.month

le_route = joblib.load('le_route.pkl')
le_weather = joblib.load('le_weather.pkl')

df['route_encoded'] = le_route.transform(df['route'])
df['weather_encoded'] = le_weather.transform(df['weather'])

features = ['route_encoded', 'weather_encoded', 'is_weekend', 'hour', 'day_of_week', 'month']
X = df[features].values
y = df['speed_kmh'].values

# Standardize features for Neural Network
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
joblib.dump(scaler_X, 'scaler_X_lstm.pkl')

# Reshape input to be 3D [samples, timesteps, features] for LSTM
# We use timestep = 1 to predict based on current hour context
X_lstm = np.reshape(X_scaled, (X_scaled.shape[0], 1, X_scaled.shape[1]))

# Split chronological
split_idx = int(len(X_lstm) * 0.8)
X_train, X_test = X_lstm[:split_idx], X_lstm[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print("Building LSTM model...")
lstm_model = Sequential([
    LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), activation='relu', return_sequences=True),
    Dropout(0.2),
    LSTM(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1) # Predict speed
])

lstm_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

print("Training LSTM... (Look at the epochs!)")
lstm_model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test))

print("Evaluating...")
loss, mae = lstm_model.evaluate(X_test, y_test)
print(f"Test RMSE: {np.sqrt(loss):.2f}")
print(f"Test MAE: {mae:.2f}")

print("Saving LSTM model...")
lstm_model.save('lstm_traffic_model.h5')
print("Xong! Model LSTM chuyên sâu đã được huấn luyện thành công.")
