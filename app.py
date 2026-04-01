import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import os

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Dự Báo Giao Thông HN", page_icon="🚦", layout="wide")

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #1E3A8A; text-align: center; }
    .sub-header { font-size: 1.2rem; color: #4B5563; text-align: center; margin-bottom: 2rem; }
    .metric-box { padding: 15px; border-radius: 10px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-header'>🚦 Hệ Thống Dự Báo Giao Thông Hà Nội</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Mô hình AI: Random Forest & Deep Learning (LSTM)</div>", unsafe_allow_html=True)

# --- LOAD THƯ VIỆN DL ---
def load_keras_model():
    try:
        from tensorflow.keras.models import load_model
        return load_model('lstm_traffic_model.h5')
    except Exception as e:
        return None

# --- LOAD MODELS VÀ DATA ---
@st.cache_resource
def load_base_models():
    try:
        rf_model = joblib.load('rf_traffic_model.pkl')
        le_route = joblib.load('le_route.pkl')
        le_weather = joblib.load('le_weather.pkl')
        return rf_model, le_route, le_weather
    except:
        return None, None, None

@st.cache_resource
def load_lstm_scaler():
    try:
        return joblib.load('scaler_X_lstm.pkl')
    except:
        return None

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('trafficstats_hanoi_mock.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except:
        return None

rf_model, le_route, le_weather = load_base_models()
lstm_scaler = load_lstm_scaler()
df = load_data()

if not rf_model or df is None:
    st.error("Vui lòng chạy `python traffic_pipeline.py` trước để sinh file Model!")
    st.stop()

# --- SIDEBAR ---
st.sidebar.header("Tùy Chỉnh Dự Báo 🛠️")

# Lựa chọn model AI
st.sidebar.subheader("🧠 Chọn Mô Hình AI")
model_options = ["Random Forest (Nhanh & Ổn định)"]
has_lstm = os.path.exists('lstm_traffic_model.h5')
if has_lstm:
    model_options.append("Deep Learning LSTM (Nâng cao)")

selected_model_type = st.sidebar.radio("Lựa chọn:", model_options)
is_using_lstm = "LSTM" in selected_model_type

if is_using_lstm:
    lstm_model = load_keras_model()
    if lstm_model is None:
        st.error("Lỗi: Không tải được mô hình LSTM, mặc dù file H5 tồn tại.")
        is_using_lstm = False

# Route
routes = le_route.classes_.tolist()
selected_route = st.sidebar.selectbox("Chọn Tuyến Đường", routes)

# Date/Time
st.sidebar.subheader("Thời Gian")
selected_date = st.sidebar.date_input("Chọn Ngày Dự Báo", datetime.today())
selected_hour = st.sidebar.slider("Chọn Giờ (0-23h)", 0, 23, 8) 

# Weather & Context
st.sidebar.subheader("Ngữ Cảnh")
weathers = le_weather.classes_.tolist()
selected_weather = st.sidebar.selectbox("Thời Tiết", weathers, index=weathers.index("Clear") if "Clear" in weathers else 0)
is_weekend = st.sidebar.checkbox("Là Ngày Cuối Tuần (T7, CN)?", value=selected_date.weekday() >= 5)

# --- XỬ LÝ DỮ LIỆU ---
pred_hour = selected_hour
pred_dayofweek = selected_date.weekday()
pred_month = selected_date.month

route_encoded = le_route.transform([selected_route])[0]
weather_encoded = le_weather.transform([selected_weather])[0]
weekend_encoded = 1 if is_weekend else 0

def predict_speed(h):
    # Dùng cho các khung giờ liên tiếp
    features = np.array([[route_encoded, weather_encoded, weekend_encoded, h, pred_dayofweek, pred_month]])
    if is_using_lstm:
        # LSTM input shape: (1, 1, 6) sau standardized
        feat_scaled = lstm_scaler.transform(features)
        feat_lstm = np.reshape(feat_scaled, (1, 1, 6))
        return lstm_model.predict(feat_lstm, verbose=0)[0][0]
    else:
        return rf_model.predict(features)[0]

predicted_speed = predict_speed(selected_hour)

if selected_route == 'Nguyen Trai': base = 40
elif selected_route == 'Vanh Dai 3': base = 60
else: base = 35

congestion_ratio = predicted_speed / base
if congestion_ratio > 0.8: status, color = "🟢 THÔNG THOÁNG", "#10B981"
elif congestion_ratio > 0.4: status, color = "🟡 ƯN ỨC / CHẬM", "#F59E0B"
else: status, color = "🔴 TẮC NGHẼN NẶNG", "#EF4444"

# --- HIỂN THỊ KẾT QUẢ HIỆN TẠI ---
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"<div class='metric-box' style='background-color: {color}20;'><h3 style='color: {color}'>Tốc Độ Dự Báo</h3><h2>{predicted_speed:.1f} km/h</h2></div>", unsafe_allow_html=True)
with col2:
    delay_min = max(0, (60 / predicted_speed) - (60 / base))
    st.markdown(f"<div class='metric-box' style='background-color: #F3F4F6;'><h3 style='color: #4B5563'>Độ Trễ / 1km</h3><h2>{delay_min:.1f} phút</h2></div>", unsafe_allow_html=True)
with col3:
    st.markdown(f"<div class='metric-box' style='background-color: {color}20;'><h3 style='color: {color}'>Tình Trạng</h3><h2>{status}</h2></div>", unsafe_allow_html=True)

st.write("---")

# --- BIỂU ĐỒ 24H ---
st.subheader(f"📈 Dự Báo Cả Ngày Cho {selected_route} Ngày {selected_date.strftime('%d/%m/%Y')} (Sử dụng: {selected_model_type})")

hours_24 = list(range(24))
predictions_24h = [predict_speed(h) for h in hours_24]

fig = go.Figure()
fig.add_trace(go.Scatter(x=hours_24, y=predictions_24h, mode='lines+markers', name='Tốc Độ (km/h)', line=dict(color='#3B82F6', width=3)))
fig.add_trace(go.Scatter(x=hours_24, y=[base]*24, mode='lines', name='Tốc Độ Chuẩn', line=dict(color='#10B981', width=2, dash='dash')))

fig.update_layout(
    xaxis_title="Giờ trong ngày (0-23h)", yaxis_title="Tốc độ (Km/h)",
    hovermode="x unified", template="plotly_white", xaxis=dict(tickmode='linear', tick0=0, dtick=1)
)
st.plotly_chart(fig, use_container_width=True)
