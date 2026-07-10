import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

def render_tab1(selected_date, selected_model_type, selected_route, selected_hour, selected_weather,
                is_weekend, weekend_encoded, pred_dayofweek, pred_month, route_encoded, weather_encoded,
                rf_model, lstm_scaler, gru_scaler, le_route, le_weather, is_using_lstm, is_using_gru,
                load_keras_model, predict_speed, get_tomtom_features, has_realtime,
                predictions_24h=None, predictions_7am=None):

    # ─────────────────────────────────────────────────────────────
    # 📋 BÁO CÁO NHANH 7:00 SÁNG  (fragment: chỉ re-render khi
    #    selected_date / selected_model_type / selected_weather đổi)
    # ─────────────────────────────────────────────────────────────
    @st.fragment
    def _render_7am_report():
        st.markdown("### 📋 Báo Cáo Giao Thông Nhanh Lúc 7:00 Sáng (Bài toán Nhị phân)")
        st.markdown(f"*Dự báo trạng thái giao thông lúc **7:00 AM** ngày **{selected_date.strftime('%d/%m/%Y')}** bằng mô hình **{selected_model_type}***")

        m_cols = st.columns(3)
        route_list = ['Nguyen Trai', 'Vanh Dai 3', 'Ton Duc Thang']

        # Tốc độ chuẩn theo nguồn dữ liệu
        base_speed = {
            'realtime': {'Nguyen Trai': 31, 'Vanh Dai 3': 41, 'Ton Duc Thang': 28},
            'mock':     {'Nguyen Trai': 40, 'Vanh Dai 3': 60, 'Ton Duc Thang': 35},
        }
        base_map = base_speed['realtime'] if has_realtime else base_speed['mock']

        for idx, r_name in enumerate(route_list):
            with m_cols[idx]:
                try:
                    # Dùng dict predictions_7am đã tính sẵn (cache từ app.py)
                    if predictions_7am and r_name in predictions_7am:
                        speed_7am = predictions_7am[r_name]
                    else:
                        # Fallback: tính lại nếu không có cache
                        r_enc = le_route.transform([r_name])[0]
                        w_enc = le_weather.transform([selected_weather])[0]
                        is_wk = 1 if selected_date.weekday() >= 5 else 0
                        ti_7, jc_7, jl_7, jd_7, _ = get_tomtom_features(selected_date, 7)
                        features_7am = np.array([[
                            r_enc, w_enc, is_wk, 7, selected_date.weekday(), selected_date.month,
                            ti_7, jc_7, jl_7, jd_7
                        ]])
                        if is_using_lstm:
                            l_model = load_keras_model('lstm')
                            if l_model is not None and lstm_scaler is not None:
                                feat_scaled = lstm_scaler.transform(features_7am)
                                feat_3d = np.reshape(feat_scaled, (1, 1, 10))
                                speed_7am = float(l_model.predict(feat_3d, verbose=0)[0][0])
                            else:
                                speed_7am = float(rf_model.predict(features_7am)[0])
                        elif is_using_gru:
                            g_model = load_keras_model('gru')
                            if g_model is not None and gru_scaler is not None:
                                feat_scaled = gru_scaler.transform(features_7am)
                                feat_3d = np.reshape(feat_scaled, (1, 1, 10))
                                speed_7am = float(g_model.predict(feat_3d, verbose=0)[0][0])
                            else:
                                speed_7am = float(rf_model.predict(features_7am)[0])
                        else:
                            speed_7am = float(rf_model.predict(features_7am)[0])

                    base_7am = base_map.get(r_name, 35)
                    ratio_7am = speed_7am / base_7am
                    if ratio_7am <= 0.7:
                        status_bin = "🔴 CÓ ÙN TẮC"
                        color_bin = "#EF4444"
                    else:
                        status_bin = "🟢 KHÔNG ÙN TẮC"
                        color_bin = "#10B981"

                    st.markdown(f"""
                    <div class='metric-box' style='background-color: #1E293B; border-top: 4px solid {color_bin}; color: #F8FAFC; padding: 15px; border-radius: 10px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.3);'>
                        <h4 style='margin: 0; color: #94A3B8; font-weight: 500;'>{r_name} (Chuẩn: {base_7am}km/h)</h4>
                        <h3 style='color: {color_bin}; margin: 10px 0; font-size: 1.5rem; font-weight: bold;'>{status_bin}</h3>
                        <p style='margin: 0; font-size: 1.25rem; font-weight: bold; color: #F1F5F9;'>{speed_7am:.1f} km/h</p>
                        <p style='margin: 0; font-size: 0.85rem; color: #94A3B8;'>Tỷ lệ tốc độ: {ratio_7am:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as ex:
                    st.error(f"Lỗi dự báo tuyến {r_name}: {ex}")

    _render_7am_report()
    st.write("---")

    # ─────────────────────────────────────────────────────────────
    # Dự báo giờ được chọn từ sidebar
    # ─────────────────────────────────────────────────────────────
    predicted_speed = predict_speed(selected_hour)

    if has_realtime:
        if selected_route == 'Nguyen Trai': base = 31
        elif selected_route == 'Vanh Dai 3': base = 41
        else: base = 28
    else:
        if selected_route == 'Nguyen Trai': base = 40
        elif selected_route == 'Vanh Dai 3': base = 60
        else: base = 35

    congestion_ratio = predicted_speed / base
    if congestion_ratio > 0.7: status, color = "🟢 THÔNG THOÁNG", "#10B981"
    elif congestion_ratio > 0.4: status, color = "🟡 ÙN ỨC / CHẬM", "#F59E0B"
    else: status, color = "🔴 TẮC NGHẼN NẶNG", "#EF4444"

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class='metric-box' style='background-color: #1E293B; border-top: 4px solid {color}; color: #F8FAFC; padding: 15px; border-radius: 10px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.3);'>
            <h3 style='color: #94A3B8; margin: 0; font-size: 1rem; font-weight: 500;'>Tốc Độ Dự Báo</h3>
            <h2 style='color: #F1F5F9; margin: 10px 0 0 0; font-size: 1.8rem; font-weight: 700;'>{predicted_speed:.1f} km/h</h2>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        delay_min = max(0, (60 / predicted_speed) - (60 / base)) if predicted_speed > 0 else 0
        st.markdown(f"""
        <div class='metric-box' style='background-color: #1E293B; border-top: 4px solid #64748B; color: #F8FAFC; padding: 15px; border-radius: 10px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.3);'>
            <h3 style='color: #94A3B8; margin: 0; font-size: 1rem; font-weight: 500;'>Độ Trễ / 1km</h3>
            <h2 style='color: #F1F5F9; margin: 10px 0 0 0; font-size: 1.8rem; font-weight: 700;'>{delay_min:.1f} phút</h2>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class='metric-box' style='background-color: #1E293B; border-top: 4px solid {color}; color: #F8FAFC; padding: 15px; border-radius: 10px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.3);'>
            <h3 style='color: #94A3B8; margin: 0; font-size: 1rem; font-weight: 500;'>Tình Trạng</h3>
            <h2 style='color: {color}; margin: 10px 0 0 0; font-size: 1.8rem; font-weight: 700;'>{status}</h2>
        </div>
        """, unsafe_allow_html=True)

    # Hiển thị bối cảnh giao thông TomTom Hà Nội sử dụng cho dự báo
    ti_sel, jc_sel, jl_sel, jd_sel, is_actual_sel = get_tomtom_features(selected_date, selected_hour)
    source_str = "Dữ liệu thực tế TomTom Index" if is_actual_sel else "Trung bình lịch sử"
    st.markdown(f"""
    <div style='background-color: #1E293B; padding: 15px; border-radius: 8px; border-left: 4px solid #3B82F6; margin-top: 15px; color: #F8FAFC; box-shadow: 0 4px 6px rgba(0,0,0,0.3);'>
        <h4 style='margin: 0 0 10px 0; color: #F1F5F9; font-weight: 600;'>🏙️ Chỉ số giao thông Hà Nội tại thời điểm dự báo</h4>
        <div style='display: flex; gap: 20px; flex-wrap: wrap; font-size: 0.95rem;'>
            <div><b>Mức độ ùn tắc (Traffic Index):</b> {ti_sel:.1f}%</div>
            <div><b>Số vụ kẹt xe:</b> {int(jc_sel)} vụ</div>
            <div><b>Tổng chiều dài kẹt xe:</b> {jl_sel:.2f} km</div>
            <div><b>Tổng thời gian trễ:</b> {jd_sel:.1f} phút</div>
            <div><b>Nguồn dữ liệu:</b> <span class="source-tag {'source-tomtom' if is_actual_sel else 'source-mock'}">{source_str}</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.write("---")

    # ─────────────────────────────────────────────────────────────
    # 📈 Biểu đồ 24h — fragment: chỉ re-render khi route/model/date/weather đổi
    # ─────────────────────────────────────────────────────────────
    @st.fragment
    def _render_chart_24h():
        st.subheader(f"📈 Dự Báo Cả Ngày Cho {selected_route} Ngày {selected_date.strftime('%d/%m/%Y')} ({selected_model_type})")

        hours_24 = list(range(24))

        # Dùng predictions_24h đã cache từ app.py nếu có
        if predictions_24h and len(predictions_24h) == 24:
            preds_list = predictions_24h
        else:
            preds_list = [predict_speed(h) for h in hours_24]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hours_24, y=preds_list,
            mode='lines+markers', name='Tốc Độ (km/h)',
            line=dict(color='#3B82F6', width=3)
        ))
        fig.add_trace(go.Scatter(
            x=hours_24, y=[base] * 24,
            mode='lines', name='Tốc Độ Chuẩn',
            line=dict(color='#10B981', width=2, dash='dash')
        ))
        fig.update_layout(
            xaxis_title="Giờ trong ngày (0-23h)", yaxis_title="Tốc độ (Km/h)",
            hovermode="x unified", template="plotly_white",
            xaxis=dict(tickmode='linear', tick0=0, dtick=1)
        )
        st.plotly_chart(fig, use_container_width=True)

    _render_chart_24h()
