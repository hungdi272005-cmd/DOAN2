import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import os
import requests

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Dự Báo Giao Thông HN", page_icon="🚦", layout="wide")

# Load API keys
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass
TOMTOM_API_KEY = os.getenv('TOMTOM_API_KEY', '') or st.secrets.get('TOMTOM_API_KEY', '')
OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY', '') or st.secrets.get('OPENWEATHER_API_KEY', '')

# Cấu hình tuyến đường
HANOI_ROUTES = {
    'Nguyen Trai': {'lat': 21.0024, 'lon': 105.7979, 'desc': 'Nguyễn Trãi - Thanh Xuân'},
    'Vanh Dai 3': {'lat': 20.9952, 'lon': 105.7872, 'desc': 'Vành Đai 3 - Linh Đàm'},
    'Ton Duc Thang': {'lat': 21.0256, 'lon': 105.8365, 'desc': 'Tôn Đức Thắng - Đống Đa'},
}

# --- STYLING (CSS) ---
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #1E3A8A; text-align: center; }
    .sub-header { font-size: 1.2rem; color: #4B5563; text-align: center; margin-bottom: 2rem; }
    .metric-box { 
        padding: 15px; 
        border-radius: 10px; 
        text-align: center; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); 
    }
    .realtime-badge { 
        display: inline-block; padding: 4px 12px; border-radius: 20px; 
        font-size: 0.8rem; font-weight: bold; animation: pulse 2s infinite;
    }
    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.6; } }
    .live-dot { color: #EF4444; font-size: 1.2rem; }
    .source-tag { 
        display: inline-block; padding: 2px 8px; border-radius: 4px; 
        font-size: 0.75rem; font-weight: bold; margin-left: 8px;
    }
    .source-tomtom { background-color: #1E3A8A; color: #F8FAFC; }
    .source-mock { background-color: #64748B; color: #F8FAFC; }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-header'>🚦 Hệ Thống Dự Báo Giao Thông Hà Nội</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>AI Models: Random Forest & Deep Learning (LSTM / GRU) | Dữ liệu thật từ TomTom</div>", unsafe_allow_html=True)

# --- LOAD MODELS & SCALERS ---
def load_keras_model(model_name):
    try:
        from tensorflow.keras.models import load_model
        keras_path = f'{model_name}_traffic_model.keras'
        h5_path = f'{model_name}_traffic_model.h5'
        if os.path.exists(keras_path):
            return load_model(keras_path)
        elif os.path.exists(h5_path):
            return load_model(h5_path)
        return None
    except Exception as e:
        return None

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

@st.cache_resource
def load_gru_scaler():
    try:
        return joblib.load('scaler_X_gru.pkl')
    except:
        return None

@st.cache_data(ttl=60)
def load_realtime_data():
    try:
        df = pd.read_csv('hanoi_traffic_realtime.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except:
        return None

@st.cache_resource
def load_tomtom_profiles():
    try:
        return joblib.load('tomtom_profiles.pkl')
    except:
        return None

@st.cache_data
def load_raw_tomtom():
    try:
        from data_utils import clean_and_load_tomtom
        df = clean_and_load_tomtom()
        df['timestamp_hour'] = df['LocalDateTime'].dt.floor('h')
        return df
    except:
        return None

rf_model, le_route, le_weather = load_base_models()
lstm_scaler = load_lstm_scaler()
gru_scaler = load_gru_scaler()
tomtom_profiles = load_tomtom_profiles()
df_tomtom_raw = load_raw_tomtom()

if not rf_model:
    st.error("Vui lòng chạy `python traffic_pipeline.py` trước để sinh file Model!")
    st.stop()

# Helper function to get TomTom Traffic Index features
def get_tomtom_features(date, h):
    try:
        if df_tomtom_raw is not None:
            dt = pd.Timestamp(date.year, date.month, date.day, h)
            match = df_tomtom_raw[df_tomtom_raw['timestamp_hour'] == dt]
            if not match.empty:
                row = match.iloc[0]
                return (
                    float(row['TrafficIndexLive']),
                    float(row['JamsCount']),
                    float(row['JamsLengthInKms']),
                    float(row['JamsDelay']),
                    True
                )
    except:
        pass
    
    try:
        if tomtom_profiles is not None:
            dow = date.weekday()
            match = tomtom_profiles[(tomtom_profiles['day_of_week'] == dow) & (tomtom_profiles['hour'] == h)]
            if not match.empty:
                row = match.iloc[0]
                return (
                    float(row['TrafficIndexLive']),
                    float(row['JamsCount']),
                    float(row['JamsLengthInKms']),
                    float(row['JamsDelay']),
                    False
                )
    except:
        pass
        
    return 30.0, 15.0, 5.0, 50.0, False

# --- SIDEBAR ---
st.sidebar.header("Tùy Chỉnh Dự Báo 🛠️")

# Data source status
st.sidebar.subheader("📁 Nguồn Dữ Liệu")
has_realtime = os.path.exists('hanoi_traffic_realtime.csv')
if has_realtime:
    df_realtime = load_realtime_data()
    realtime_count = len(df_realtime) if df_realtime is not None else 0
    st.sidebar.success(f"✅ Có {realtime_count} records dữ liệu thật")
else:
    st.sidebar.info("💡 Chạy `python tomtom_collector.py --schedule` để thu thập dữ liệu thật")

# Model selection
st.sidebar.subheader("🧠 Chọn Mô Hình AI")
model_options = ["Random Forest (Nhanh & Ổn định)"]
has_lstm = os.path.exists('lstm_traffic_model.keras') or os.path.exists('lstm_traffic_model.h5')
has_gru = os.path.exists('gru_traffic_model.keras') or os.path.exists('gru_traffic_model.h5')
if has_lstm:
    model_options.append("Deep Learning LSTM (Nâng cao)")
if has_gru:
    model_options.append("Deep Learning GRU (Nhanh & Chính xác)")

selected_model_type = st.sidebar.radio("Lựa chọn:", model_options)
is_using_lstm = "LSTM" in selected_model_type
is_using_gru = "GRU" in selected_model_type

lstm_model = None
gru_model = None

if is_using_lstm:
    lstm_model = load_keras_model('lstm')
    if lstm_model is None:
        st.error("Lỗi: Không tải được mô hình LSTM.")
        is_using_lstm = False

if is_using_gru:
    gru_model = load_keras_model('gru')
    if gru_model is None:
        st.error("Lỗi: Không tải được mô hình GRU.")
        is_using_gru = False

# Route
routes = le_route.classes_.tolist()
selected_route = st.sidebar.selectbox("Chọn Tuyến Đường", routes)

# Date/Time
st.sidebar.subheader("Thời Gian")
selected_date = st.sidebar.date_input("Chọn Ngày Dự Báo", datetime.today())
selected_hour = st.sidebar.slider("Chọn Giờ (0-23h)", 0, 23, 8) 

# Weather
st.sidebar.subheader("Ngữ Cảnh")
weathers = le_weather.classes_.tolist()
selected_weather = st.sidebar.selectbox("Thời Tiết", weathers, index=weathers.index("Clear") if "Clear" in weathers else 0)
is_weekend = selected_date.weekday() >= 5

# --- PREDICTION FUNCTION ---
pred_hour = selected_hour
pred_dayofweek = selected_date.weekday()
pred_month = selected_date.month

route_encoded = le_route.transform([selected_route])[0]
weather_encoded = le_weather.transform([selected_weather])[0]
weekend_encoded = 1 if is_weekend else 0

def predict_speed(h):
    ti, jc, jl, jd, is_actual = get_tomtom_features(selected_date, h)
    features = np.array([[
        route_encoded, weather_encoded, weekend_encoded, h, pred_dayofweek, pred_month,
        ti, jc, jl, jd
    ]])
    if is_using_lstm and lstm_model is not None:
        feat_scaled = lstm_scaler.transform(features)
        feat_3d = np.reshape(feat_scaled, (1, 1, 10))
        return float(lstm_model.predict(feat_3d, verbose=0)[0][0])
    elif is_using_gru and gru_model is not None:
        feat_scaled = gru_scaler.transform(features)
        feat_3d = np.reshape(feat_scaled, (1, 1, 10))
        return float(gru_model.predict(feat_3d, verbose=0)[0][0])
    else:
        return rf_model.predict(features)[0]

def predict_speed_for_route(route_name, h):
    """Dự báo tốc độ cho bất kỳ tuyến đường nào theo giờ."""
    r_encoded = le_route.transform([route_name])[0]
    ti, jc, jl, jd, is_actual = get_tomtom_features(selected_date, h)
    features = np.array([[
        r_encoded, weather_encoded, weekend_encoded, h, pred_dayofweek, pred_month,
        ti, jc, jl, jd
    ]])
    if is_using_lstm and lstm_model is not None:
        feat_scaled = lstm_scaler.transform(features)
        feat_3d = np.reshape(feat_scaled, (1, 1, 10))
        return float(lstm_model.predict(feat_3d, verbose=0)[0][0])
    elif is_using_gru and gru_model is not None:
        feat_scaled = gru_scaler.transform(features)
        feat_3d = np.reshape(feat_scaled, (1, 1, 10))
        return float(gru_model.predict(feat_3d, verbose=0)[0][0])
    else:
        return float(rf_model.predict(features)[0])

# --- CACHED BATCH PREDICTIONS (tránh tính lại khi user chỉ đổi tab) ---
# NOTE: Không truyền model object vào @st.cache_data vì không pickle được.
# Thay vào đó, cache key là các tham số số đơn giản + model_type string.
@st.cache_data(show_spinner=False)
def cached_predict_24h(route_enc, weather_enc, weekend_enc, dayofweek, month,
                       model_type, date_str):
    """Cache dự báo 24h cho 1 tuyến đường.
    Model được lấy từ @st.cache_resource — an toàn, không cần pickle."""
    # Lấy model từ cache resource (sỚ) thay vì nhận tham số
    _rf, _le_r, _le_w = load_base_models()
    _lstm_sc = load_lstm_scaler()
    _gru_sc  = load_gru_scaler()
    _lstm_m  = load_keras_model('lstm') if 'LSTM' in model_type else None
    _gru_m   = load_keras_model('gru')  if 'GRU'  in model_type else None

    date_obj = pd.Timestamp(date_str).date()
    
    # 1. Build features batch
    features_list = []
    for h in range(24):
        try:
            ti, jc, jl, jd, _ = get_tomtom_features(date_obj, h)
            features_list.append([route_enc, weather_enc, weekend_enc, h, dayofweek, month, ti, jc, jl, jd])
        except Exception:
            # Fallback values if API fails
            features_list.append([route_enc, weather_enc, weekend_enc, h, dayofweek, month, 0, 0, 0, 0])
            
    features = np.array(features_list)
    
    # 2. Predict batch
    try:
        if 'LSTM' in model_type and _lstm_m is not None and _lstm_sc is not None:
            feat_scaled = _lstm_sc.transform(features)
            feat_3d = np.reshape(feat_scaled, (24, 1, 10))
            speeds = _lstm_m.predict(feat_3d, verbose=0).flatten()
        elif 'GRU' in model_type and _gru_m is not None and _gru_sc is not None:
            feat_scaled = _gru_sc.transform(features)
            feat_3d = np.reshape(feat_scaled, (24, 1, 10))
            speeds = _gru_m.predict(feat_3d, verbose=0).flatten()
        else:
            speeds = _rf.predict(features)
        
        return [float(s) for s in speeds]
    except Exception:
        return [30.0] * 24


@st.cache_data(show_spinner=False)
def cached_predict_all_routes_next_hours(weather_enc, weekend_enc, dayofweek, month,
                                          model_type, date_str, current_hour,
                                          route_names_tuple):
    """Cache dự báo 6h tới cho TẤT CẢ tuyến đường (batch)."""
    _rf, _le_r, _le_w = load_base_models()
    _lstm_sc = load_lstm_scaler()
    _gru_sc  = load_gru_scaler()
    _lstm_m  = load_keras_model('lstm') if 'LSTM' in model_type else None
    _gru_m   = load_keras_model('gru')  if 'GRU'  in model_type else None

    date_obj = pd.Timestamp(date_str).date()
    hours = list(range(current_hour, min(current_hour + 7, 24)))

    # 1. Build features batch
    features_list = []
    meta_info = [] # store (route_name, hour) to map back results
    for route_name in route_names_tuple:
        r_enc = _le_r.transform([route_name])[0]
        for h in hours:
            try:
                ti, jc, jl, jd, _ = get_tomtom_features(date_obj, h)
                features_list.append([r_enc, weather_enc, weekend_enc, h, dayofweek, month, ti, jc, jl, jd])
            except Exception:
                features_list.append([r_enc, weather_enc, weekend_enc, h, dayofweek, month, 0, 0, 0, 0])
            meta_info.append((route_name, h))
            
    if not features_list:
        return {r: [] for r in route_names_tuple}

    features = np.array(features_list)
    
    # 2. Predict batch
    try:
        if 'LSTM' in model_type and _lstm_m is not None and _lstm_sc is not None:
            feat_scaled = _lstm_sc.transform(features)
            feat_3d = np.reshape(feat_scaled, (len(features), 1, 10))
            speeds = _lstm_m.predict(feat_3d, verbose=0).flatten()
        elif 'GRU' in model_type and _gru_m is not None and _gru_sc is not None:
            feat_scaled = _gru_sc.transform(features)
            feat_3d = np.reshape(feat_scaled, (len(features), 1, 10))
            speeds = _gru_m.predict(feat_3d, verbose=0).flatten()
        else:
            speeds = _rf.predict(features)
    except Exception:
        speeds = [30.0] * len(features)

    # 3. Map back to dictionary
    all_predictions = {r: [] for r in route_names_tuple}
    for (route_name, h), speed in zip(meta_info, speeds):
        all_predictions[route_name].append({'hour': h, 'speed': round(float(speed), 1)})
        
    return all_predictions

@st.cache_data(show_spinner=False)
def cached_predict_7am_all_routes(weather_enc, weekend_enc, dayofweek, month,
                                   model_type, date_str, route_names_tuple):
    """Cache dự báo 7:00 sáng cho tất cả tuyến (batch)."""
    _rf, _le_r, _le_w = load_base_models()
    _lstm_sc = load_lstm_scaler()
    _gru_sc  = load_gru_scaler()
    _lstm_m  = load_keras_model('lstm') if 'LSTM' in model_type else None
    _gru_m   = load_keras_model('gru')  if 'GRU'  in model_type else None

    date_obj = pd.Timestamp(date_str).date()
    
    features_list = []
    meta_info = []
    for route_name in route_names_tuple:
        r_enc = _le_r.transform([route_name])[0]
        try:
            ti, jc, jl, jd, _ = get_tomtom_features(date_obj, 7)
            features_list.append([r_enc, weather_enc, weekend_enc, 7, dayofweek, month, ti, jc, jl, jd])
        except Exception:
            features_list.append([r_enc, weather_enc, weekend_enc, 7, dayofweek, month, 0, 0, 0, 0])
        meta_info.append(route_name)
            
    if not features_list:
        return {}

    features = np.array(features_list)
    
    try:
        if 'LSTM' in model_type and _lstm_m is not None and _lstm_sc is not None:
            feat_scaled = _lstm_sc.transform(features)
            feat_3d = np.reshape(feat_scaled, (len(features), 1, 10))
            speeds = _lstm_m.predict(feat_3d, verbose=0).flatten()
        elif 'GRU' in model_type and _gru_m is not None and _gru_sc is not None:
            feat_scaled = _gru_sc.transform(features)
            feat_3d = np.reshape(feat_scaled, (len(features), 1, 10))
            speeds = _gru_m.predict(feat_3d, verbose=0).flatten()
        else:
            speeds = _rf.predict(features)
    except Exception:
        speeds = [30.0] * len(features)

    results = {}
    for route_name, speed in zip(meta_info, speeds):
        results[route_name] = float(speed)
        
    return results

# --- API HELPER FUNCTIONS ---
@st.cache_data(ttl=60)
def fetch_tomtom_flow(lat, lon):
    try:
        url = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
        params = {'key': TOMTOM_API_KEY, 'point': f'{lat},{lon}', 'unit': 'KMPH'}
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        flow = response.json()['flowSegmentData']
        return {
            'current_speed': flow['currentSpeed'],
            'free_flow_speed': flow['freeFlowSpeed'],
            'current_travel_time': flow['currentTravelTime'],
            'free_flow_travel_time': flow['freeFlowTravelTime'],
            'confidence': flow['confidence'],
            'road_closure': flow.get('roadClosure', False),
        }
    except:
        return None

@st.cache_data(ttl=300)
def fetch_weather():
    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            'lat': 21.0285, 'lon': 105.8542,
            'appid': OPENWEATHER_API_KEY,
            'units': 'metric', 'lang': 'vi'
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return {
            'desc': data['weather'][0]['description'],
            'temp': data['main']['temp'],
            'humidity': data['main']['humidity'],
            'icon': data['weather'][0]['icon']
        }
    except:
        return None

# === PRE-COMPUTE PREDICTIONS (1 LẦN, CACHE LẠI) ===
from datetime import datetime as _dt
_current_hour = _dt.now().hour
_date_str = str(selected_date)          # dùng str làm cache key
_route_names_tuple = tuple(HANOI_ROUTES.keys())

# Dự báo 24h của tuyến đường được chọn (Tab 1)
predictions_24h = cached_predict_24h(
    route_encoded, weather_encoded, weekend_encoded,
    pred_dayofweek, pred_month,
    selected_model_type, _date_str
)

# Dự báo 7h sáng tất cả tuyến (Tab 1 - báo cáo nhanh)
predictions_7am = cached_predict_7am_all_routes(
    weather_encoded, weekend_encoded,
    pred_dayofweek, pred_month,
    selected_model_type, _date_str, _route_names_tuple
)

# Dự báo 6h tới cho 3 tuyến (Tab 3 - bản đồ)
all_route_predictions = cached_predict_all_routes_next_hours(
    weather_encoded, weekend_encoded,
    pred_dayofweek, pred_month,
    selected_model_type, _date_str, _current_hour, _route_names_tuple
)

# === TABS ROUTING ===
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Dự Báo AI", "🔴 Real-time", "🗺️ Bản Đồ Traffic", "📈 Lịch Sử", "🎯 Đánh Giá Mô Hình"])

# Load Tab Views
from views.tab1_prediction import render_tab1
from views.tab2_realtime import render_tab2
from views.tab3_map import render_tab3
from views.tab4_history import render_tab4
from views.tab5_evaluation import render_tab5

with tab1:
    render_tab1(
        selected_date, selected_model_type, selected_route, selected_hour, selected_weather,
        is_weekend, weekend_encoded, pred_dayofweek, pred_month, route_encoded, weather_encoded,
        rf_model, lstm_scaler, gru_scaler, le_route, le_weather, is_using_lstm, is_using_gru,
        load_keras_model, predict_speed, get_tomtom_features, has_realtime,
        predictions_24h=predictions_24h,
        predictions_7am=predictions_7am
    )

with tab2:
    render_tab2(TOMTOM_API_KEY, HANOI_ROUTES, fetch_tomtom_flow, fetch_weather)

with tab3:
    render_tab3(TOMTOM_API_KEY, HANOI_ROUTES, fetch_tomtom_flow,
                predict_speed_for_route, selected_date, selected_model_type,
                all_predictions=all_route_predictions)

with tab4:
    render_tab4(load_realtime_data)

with tab5:
    render_tab5(
        selected_model_type, selected_date, selected_weather, le_route, le_weather,
        rf_model, lstm_scaler, gru_scaler, load_keras_model, get_tomtom_features, has_realtime
    )
