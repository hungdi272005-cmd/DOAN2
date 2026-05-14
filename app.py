import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
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

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #1E3A8A; text-align: center; }
    .sub-header { font-size: 1.2rem; color: #4B5563; text-align: center; margin-bottom: 2rem; }
    .metric-box { padding: 15px; border-radius: 10px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .realtime-badge { 
        display: inline-block; padding: 4px 12px; border-radius: 20px; 
        font-size: 0.8rem; font-weight: bold; animation: pulse 2s infinite;
    }
    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.6; } }
    .live-dot { color: #EF4444; font-size: 1.2rem; }
    .source-tag { 
        display: inline-block; padding: 2px 8px; border-radius: 4px; 
        font-size: 0.7rem; font-weight: bold; margin-left: 8px;
    }
    .source-tomtom { background-color: #DBEAFE; color: #1E40AF; }
    .source-mock { background-color: #FEF3C7; color: #92400E; }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-header'>🚦 Hệ Thống Dự Báo Giao Thông Hà Nội</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>AI Models: Random Forest & Deep Learning (LSTM / GRU) | Dữ liệu thật từ TomTom</div>", unsafe_allow_html=True)

# --- LOAD MODELS ---
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
        st.warning(f"Chi tiết lỗi load {model_name.upper()}: {e}")
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

@st.cache_data
def load_mock_data():
    try:
        df = pd.read_csv('trafficstats_hanoi_mock.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
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

rf_model, le_route, le_weather = load_base_models()
lstm_scaler = load_lstm_scaler()
gru_scaler = load_gru_scaler()
df_mock = load_mock_data()

if not rf_model or df_mock is None:
    st.error("Vui lòng chạy `python traffic_pipeline.py` trước để sinh file Model!")
    st.stop()

# --- API HELPER FUNCTIONS ---
def fetch_tomtom_flow(lat, lon):
    """Gọi TomTom API lấy dữ liệu real-time."""
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

def fetch_weather():
    """Gọi OpenWeatherMap API lấy thời tiết."""
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

# --- SIDEBAR ---
st.sidebar.header("Tùy Chỉnh Dự Báo 🛠️")

# Data source
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
    features = np.array([[route_encoded, weather_encoded, weekend_encoded, h, pred_dayofweek, pred_month]])
    if is_using_lstm and lstm_model is not None:
        feat_scaled = lstm_scaler.transform(features)
        feat_3d = np.reshape(feat_scaled, (1, 1, 6))
        return float(lstm_model.predict(feat_3d, verbose=0)[0][0])
    elif is_using_gru and gru_model is not None:
        feat_scaled = gru_scaler.transform(features)
        feat_3d = np.reshape(feat_scaled, (1, 1, 6))
        return float(gru_model.predict(feat_3d, verbose=0)[0][0])
    else:
        return rf_model.predict(features)[0]

# === TABS ===
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dự Báo AI", "🔴 Real-time", "🗺️ Bản Đồ Traffic", "📈 Lịch Sử"])

# ============================================================
# TAB 1: DỰ BÁO AI (giữ nguyên logic cũ + nâng cấp giao diện)
# ============================================================
with tab1:
    predicted_speed = predict_speed(selected_hour)

    if selected_route == 'Nguyen Trai': base = 40
    elif selected_route == 'Vanh Dai 3': base = 60
    else: base = 35

    congestion_ratio = predicted_speed / base
    if congestion_ratio > 0.8: status, color = "🟢 THÔNG THOÁNG", "#10B981"
    elif congestion_ratio > 0.4: status, color = "🟡 ÙN ỨC / CHẬM", "#F59E0B"
    else: status, color = "🔴 TẮC NGHẼN NẶNG", "#EF4444"

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"<div class='metric-box' style='background-color: {color}20;'><h3 style='color: {color}'>Tốc Độ Dự Báo</h3><h2>{predicted_speed:.1f} km/h</h2></div>", unsafe_allow_html=True)
    with col2:
        delay_min = max(0, (60 / predicted_speed) - (60 / base)) if predicted_speed > 0 else 0
        st.markdown(f"<div class='metric-box' style='background-color: #F3F4F6;'><h3 style='color: #4B5563'>Độ Trễ / 1km</h3><h2>{delay_min:.1f} phút</h2></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric-box' style='background-color: {color}20;'><h3 style='color: {color}'>Tình Trạng</h3><h2>{status}</h2></div>", unsafe_allow_html=True)

    st.write("---")

    st.subheader(f"📈 Dự Báo Cả Ngày Cho {selected_route} Ngày {selected_date.strftime('%d/%m/%Y')} ({selected_model_type})")

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

# ============================================================
# TAB 2: REAL-TIME DATA TỪ TOMTOM
# ============================================================
with tab2:
    st.subheader("🔴 Dữ Liệu Giao Thông Real-time")
    
    if not TOMTOM_API_KEY:
        st.error("⚠️ Chưa cấu hình TOMTOM_API_KEY trong file .env")
    else:
        st.markdown("<span class='live-dot'>●</span> <b>LIVE</b> - Dữ liệu trực tiếp từ TomTom Traffic API", unsafe_allow_html=True)
        
        if st.button("🔄 Cập Nhật Dữ Liệu", key="refresh_realtime"):
            st.cache_data.clear()
        
        # Lấy dữ liệu real-time cho tất cả tuyến
        cols = st.columns(len(HANOI_ROUTES))
        
        for idx, (route_name, route_info) in enumerate(HANOI_ROUTES.items()):
            with cols[idx]:
                flow = fetch_tomtom_flow(route_info['lat'], route_info['lon'])
                
                if flow:
                    ratio = flow['current_speed'] / flow['free_flow_speed'] if flow['free_flow_speed'] > 0 else 1
                    if ratio > 0.8:
                        status_rt = "🟢 Thông thoáng"
                        color_rt = "#10B981"
                    elif ratio > 0.5:
                        status_rt = "🟡 Ùn ứ"
                        color_rt = "#F59E0B"
                    else:
                        status_rt = "🔴 Tắc nghẽn"
                        color_rt = "#EF4444"
                    
                    st.markdown(f"""
                    <div class='metric-box' style='background-color: {color_rt}15; border-left: 4px solid {color_rt};'>
                        <h4>{route_info['desc']}</h4>
                        <h2 style='color: {color_rt};'>{flow['current_speed']} km/h</h2>
                        <p>Tự do: {flow['free_flow_speed']} km/h</p>
                        <p><b>{status_rt}</b></p>
                        <p style='font-size: 0.8rem; color: #6B7280;'>
                            Confidence: {flow['confidence']:.0%} | 
                            Travel: {flow['current_travel_time']}s (chuẩn: {flow['free_flow_travel_time']}s)
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning(f"Không lấy được dữ liệu cho {route_name}")
        
        # Gauge charts
        st.write("---")
        st.subheader("📊 Mức Tải Giao Thông")
        
        gauge_cols = st.columns(len(HANOI_ROUTES))
        for idx, (route_name, route_info) in enumerate(HANOI_ROUTES.items()):
            with gauge_cols[idx]:
                flow = fetch_tomtom_flow(route_info['lat'], route_info['lon'])
                if flow:
                    ratio_pct = (flow['current_speed'] / flow['free_flow_speed'] * 100) if flow['free_flow_speed'] > 0 else 100
                    
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=flow['current_speed'],
                        delta={'reference': flow['free_flow_speed'], 'increasing': {'color': '#10B981'}, 'decreasing': {'color': '#EF4444'}},
                        title={'text': route_name},
                        gauge={
                            'axis': {'range': [0, flow['free_flow_speed'] * 1.2]},
                            'bar': {'color': "#3B82F6"},
                            'steps': [
                                {'range': [0, flow['free_flow_speed'] * 0.4], 'color': '#FEE2E2'},
                                {'range': [flow['free_flow_speed'] * 0.4, flow['free_flow_speed'] * 0.7], 'color': '#FEF3C7'},
                                {'range': [flow['free_flow_speed'] * 0.7, flow['free_flow_speed'] * 1.2], 'color': '#D1FAE5'},
                            ],
                            'threshold': {
                                'line': {'color': '#10B981', 'width': 3},
                                'thickness': 0.8,
                                'value': flow['free_flow_speed']
                            }
                        }
                    ))
                    fig_gauge.update_layout(height=250, margin=dict(t=40, b=0, l=20, r=20))
                    st.plotly_chart(fig_gauge, use_container_width=True)

        # Thời tiết
        st.write("---")
        weather = fetch_weather()
        if weather:
            st.subheader("🌤️ Thời Tiết Hà Nội Hiện Tại")
            w_col1, w_col2, w_col3 = st.columns(3)
            with w_col1:
                st.metric("Nhiệt độ", f"{weather['temp']}°C")
            with w_col2:
                st.metric("Độ ẩm", f"{weather['humidity']}%")
            with w_col3:
                st.metric("Mô tả", weather['desc'].capitalize())
        
        st.caption(f"⏰ Cập nhật lúc: {datetime.now().strftime('%H:%M:%S %d/%m/%Y')}")

# ============================================================
# TAB 3: BẢN ĐỒ TRAFFIC
# ============================================================
with tab3:
    st.subheader("🗺️ Bản Đồ Giao Thông Hà Nội")
    
    if not TOMTOM_API_KEY:
        st.error("⚠️ Cần TOMTOM_API_KEY để hiển thị bản đồ")
    else:
        # Build markers data for JS
        markers_js = ""
        for route_name, route_info in HANOI_ROUTES.items():
            flow = fetch_tomtom_flow(route_info['lat'], route_info['lon'])
            if flow:
                ratio = flow['current_speed'] / flow['free_flow_speed'] if flow['free_flow_speed'] > 0 else 1
                if ratio > 0.8:
                    marker_color = '#10B981'
                elif ratio > 0.5:
                    marker_color = '#F59E0B'
                else:
                    marker_color = '#EF4444'
                markers_js += f"""
                var marker_{route_name.replace(' ', '')} = new tt.Marker({{
                    element: createMarkerElement('{marker_color}')
                }})
                .setLngLat([{route_info['lon']}, {route_info['lat']}])
                .setPopup(new tt.Popup({{offset: 30}}).setHTML(
                    '<h3>{route_info["desc"]}</h3>' +
                    '<p><b>Tốc độ:</b> {flow["current_speed"]} km/h</p>' +
                    '<p><b>Tự do:</b> {flow["free_flow_speed"]} km/h</p>' +
                    '<p><b>Tỷ lệ:</b> {ratio:.0%}</p>'
                ))
                .addTo(map);
                """
        
        map_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <link rel="stylesheet" type="text/css" href="https://api.tomtom.com/maps-sdk-for-web/cdn/6.x/6.25.0/maps/maps.css"/>
            <script src="https://api.tomtom.com/maps-sdk-for-web/cdn/6.x/6.25.0/maps/maps-web.min.js"></script>
            <style>
                #map {{ width: 100%; height: 550px; border-radius: 12px; }}
                .custom-marker {{
                    width: 20px; height: 20px; border-radius: 50%;
                    border: 3px solid white; box-shadow: 0 2px 6px rgba(0,0,0,0.3);
                    cursor: pointer;
                }}
            </style>
        </head>
        <body>
            <div id="map"></div>
            <script>
                function createMarkerElement(color) {{
                    var el = document.createElement('div');
                    el.className = 'custom-marker';
                    el.style.backgroundColor = color;
                    return el;
                }}
                
                var map = tt.map({{
                    key: '{TOMTOM_API_KEY}',
                    container: 'map',
                    center: [105.8342, 21.0278],
                    zoom: 12,
                    style: 'https://api.tomtom.com/style/1/style/22.2.1-*?map=2/basic_street-light&traffic_incidents=2/incidents_light&traffic_flow=2/flow_light&poi=2/poi_light'
                }});

                map.on('load', function() {{
                    // Thêm layer Traffic Flow
                    map.showTrafficFlow();
                    map.showTrafficIncidents();
                    
                    // Thêm markers cho các tuyến đường
                    {markers_js}
                }});
            </script>
        </body>
        </html>
        """
        
        st.components.v1.html(map_html, height=570)
        
        st.info("💡 **Chú thích màu:** 🟢 Xanh = Thông thoáng | 🟡 Vàng = Ùn ứ | 🔴 Đỏ = Tắc nghẽn")
        st.caption("Bản đồ sử dụng TomTom Maps SDK với Traffic Flow Tiles real-time")

# ============================================================
# TAB 4: LỊCH SỬ DỮ LIỆU
# ============================================================
with tab4:
    st.subheader("📈 Lịch Sử Dữ Liệu Thu Thập")
    
    df_rt = load_realtime_data()
    
    if df_rt is not None and len(df_rt) > 0:
        st.success(f"📊 Tổng: **{len(df_rt)}** records dữ liệu thật | "
                   f"Từ: **{df_rt['timestamp'].min().strftime('%d/%m/%Y %H:%M')}** → "
                   f"**{df_rt['timestamp'].max().strftime('%d/%m/%Y %H:%M')}**")
        
        # Filter by route
        hist_route = st.selectbox("Tuyến đường:", ['Tất cả'] + df_rt['route'].unique().tolist(), key='hist_route')
        
        if hist_route != 'Tất cả':
            df_filtered = df_rt[df_rt['route'] == hist_route]
        else:
            df_filtered = df_rt
        
        # Chart: Speed over time
        fig_hist = px.line(df_filtered, x='timestamp', y='speed_kmh', color='route',
                          title='Tốc Độ Giao Thông Theo Thời Gian (Dữ liệu thật)',
                          labels={'speed_kmh': 'Tốc độ (km/h)', 'timestamp': 'Thời gian', 'route': 'Tuyến đường'})
        fig_hist.update_layout(template='plotly_white', hovermode='x unified')
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Chart: Congestion ratio over time
        if 'congestion_ratio' in df_filtered.columns:
            fig_cong = px.line(df_filtered, x='timestamp', y='congestion_ratio', color='route',
                              title='Tỷ Lệ Tắc Nghẽn Theo Thời Gian',
                              labels={'congestion_ratio': 'Congestion Ratio', 'timestamp': 'Thời gian', 'route': 'Tuyến đường'})
            fig_cong.add_hline(y=0.8, line_dash="dash", line_color="#10B981", annotation_text="Thông thoáng")
            fig_cong.add_hline(y=0.5, line_dash="dash", line_color="#F59E0B", annotation_text="Ùn ứ")
            fig_cong.update_layout(template='plotly_white', hovermode='x unified')
            st.plotly_chart(fig_cong, use_container_width=True)
        
        # Stats
        st.subheader("📋 Thống Kê Tóm Tắt")
        stats = df_rt.groupby('route').agg({
            'speed_kmh': ['mean', 'min', 'max', 'std'],
        }).round(2)
        stats.columns = ['TB (km/h)', 'Min (km/h)', 'Max (km/h)', 'Std (km/h)']
        st.dataframe(stats, use_container_width=True)
        
        # Raw data
        with st.expander("📄 Xem dữ liệu thô"):
            st.dataframe(df_rt.tail(50), use_container_width=True)
    else:
        st.info("📭 Chưa có dữ liệu thật. Chạy lệnh sau để bắt đầu thu thập:")
        st.code("python tomtom_collector.py --schedule", language="bash")
        st.markdown("""
        **Các lệnh khác:**
        - `python tomtom_collector.py` - Thu thập 1 lần (test)
        - `python tomtom_collector.py --schedule` - Chạy liên tục mỗi 15 phút
        - `python tomtom_collector.py --schedule --interval 5` - Mỗi 5 phút
        """)
