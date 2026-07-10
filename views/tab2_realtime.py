import streamlit as st
import plotly.graph_objects as go
from datetime import datetime

def render_tab2(TOMTOM_API_KEY, HANOI_ROUTES, fetch_tomtom_flow, fetch_weather):
    st.subheader("🔴 Dữ Liệu Giao Thông Real-time")

    if not TOMTOM_API_KEY:
        st.error("⚠️ Chưa cấu hình TOMTOM_API_KEY trong file .env")
        return

    # ─────────────────────────────────────────────────────────────
    # @st.fragment: khi nhấn "Cập Nhật" chỉ re-run phần này,
    # không re-run toàn bộ app.py
    # ─────────────────────────────────────────────────────────────
    @st.fragment
    def _realtime_panel():
        st.markdown("<span class='live-dot'>●</span> <b>LIVE</b> - Dữ liệu trực tiếp từ TomTom Traffic API", unsafe_allow_html=True)

        if st.button("🔄 Cập Nhật Dữ Liệu", key="refresh_realtime"):
            # Xóa cache API để lấy dữ liệu mới
            fetch_tomtom_flow.clear()
            fetch_weather.clear()
            st.rerun(scope="fragment")

        # Lấy dữ liệu flow song song (Concurrent Fetching)
        from concurrent.futures import ThreadPoolExecutor

        routes_list = list(HANOI_ROUTES.items())

        def get_flow_wrapper(item):
            name, info = item
            return name, fetch_tomtom_flow(info['lat'], info['lon'])

        with ThreadPoolExecutor(max_workers=len(HANOI_ROUTES)) as executor:
            flow_results = dict(executor.map(get_flow_wrapper, routes_list))

        # Hiển thị Metric Cards
        cols = st.columns(len(HANOI_ROUTES))
        for idx, (route_name, route_info) in enumerate(HANOI_ROUTES.items()):
            with cols[idx]:
                flow = flow_results.get(route_name)

                if flow:
                    ratio = flow['current_speed'] / flow['free_flow_speed'] if flow['free_flow_speed'] > 0 else 1
                    if ratio > 0.7:
                        status_rt = "🟢 Thông thoáng"
                        color_rt = "#10B981"
                    elif ratio > 0.4:
                        status_rt = "🟡 Ùn ứ"
                        color_rt = "#F59E0B"
                    else:
                        status_rt = "🔴 Tắc nghẽn"
                        color_rt = "#EF4444"

                    st.markdown(f"""
                    <div class='metric-box' style='background-color: #1E293B; border-left: 4px solid {color_rt}; color: #F8FAFC; padding: 15px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.3);'>
                        <h4 style='margin: 0; color: #94A3B8; font-weight: 500;'>{route_info['desc']}</h4>
                        <h2 style='color: {color_rt}; margin: 10px 0; font-size: 1.8rem; font-weight: 700;'>{flow['current_speed']} km/h</h2>
                        <p style='margin: 0; font-size: 1.05rem;'>Tự do: {flow['free_flow_speed']} km/h</p>
                        <p style='margin: 5px 0; font-weight: bold; color: {color_rt};'>{status_rt}</p>
                        <p style='font-size: 0.8rem; color: #94A3B8; margin: 5px 0 0 0;'>
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
                flow = flow_results.get(route_name)
                if flow:
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

    _realtime_panel()
