import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

print("Loading data...")
df = pd.read_csv('trafficstats_hanoi_mock.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

print("Feature Engineering...")
# Extract time features
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['month'] = df['timestamp'].dt.month

# Encode categorical features
le_route = LabelEncoder()
df['route_encoded'] = le_route.fit_transform(df['route'])
joblib.dump(le_route, 'le_route.pkl')

le_weather = LabelEncoder()
df['weather_encoded'] = le_weather.fit_transform(df['weather'])
joblib.dump(le_weather, 'le_weather.pkl')

# Define features and target
features = ['route_encoded', 'weather_encoded', 'is_weekend', 'hour', 'day_of_week', 'month']
target = 'speed_kmh' # We predict speed, then we can derive congestion

X = df[features]
y = df[target]

print("Splitting data...")
# For time series, we shouldn't purely shuffle. We'll use chronological split.
split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

print("Training Random Forest model...")
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

print("Evaluating...")
predictions = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
mae = mean_absolute_error(y_test, predictions)

print(f"Test RMSE (km/h): {rmse:.2f}")
print(f"Test MAE (km/h): {mae:.2f}")

print("Saving model...")
joblib.dump(model, 'rf_traffic_model.pkl')
print("Pipeline completed successfully!")
