import streamlit as st
import json
from datetime import datetime

# Tọa độ polyline gần đúng cho 3 tuyến đường đã train
ROUTE_POLYLINES = {
    'Nguyen Trai': [
        [21.0105, 105.8195], [21.0090, 105.8175], [21.0075, 105.8145],
        [21.0060, 105.8110], [21.0045, 105.8060], [21.0030, 105.8020],
        [21.0024, 105.7979], [21.0010, 105.7940], [20.9990, 105.7890],
        [20.9970, 105.7850], [20.9950, 105.7800], [20.9930, 105.7760]
    ],
    'Vanh Dai 3': [
        [21.0080, 105.7990], [21.0055, 105.7965], [21.0030, 105.7940],
        [21.0005, 105.7915], [20.9980, 105.7895], [20.9952, 105.7872],
        [20.9925, 105.7850], [20.9900, 105.7830], [20.9870, 105.7805],
        [20.9845, 105.7785], [20.9820, 105.7760]
    ],
    'Ton Duc Thang': [
        [21.0330, 105.8400], [21.0315, 105.8395], [21.0300, 105.8388],
        [21.0285, 105.8380], [21.0270, 105.8373], [21.0256, 105.8365],
        [21.0240, 105.8358], [21.0225, 105.8350], [21.0210, 105.8343],
        [21.0195, 105.8335], [21.0180, 105.8328]
    ]
}


def get_speed_status(speed):
    """Trả về (trạng thái, màu, emoji) dựa trên tốc độ dự báo."""
    if speed >= 30:
        return 'Thông thoáng', '#10B981', '🟢'
    elif speed >= 20:
        return 'Ùn ứ nhẹ', '#F59E0B', '🟡'
    else:
        return 'Tắc nghẽn', '#EF4444', '🔴'


def render_tab3(TOMTOM_API_KEY, HANOI_ROUTES, fetch_tomtom_flow,
                predict_speed_for_route, selected_date, selected_model_type,
                all_predictions=None):
    st.subheader("🗺️ Bản Đồ Giao Thông Hà Nội")

    current_hour = datetime.now().hour
    route_names = list(HANOI_ROUTES.keys())
    route_descs = {name: info['desc'] for name, info in HANOI_ROUTES.items()}

    # ═══════════════════════════════════════════════════════
    # SECTION 1: Tra Tuyến Đường
    # ═══════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 🔍 Tra Tuyến Đường")

    selected_route = st.selectbox(
        "Chọn tuyến đường đã train:",
        route_names,
        format_func=lambda x: f"📍 {route_descs[x]}",
        key="tab3_route_lookup"
    )

    # ═══════════════════════════════════════════════════════
    # Fetch real-time flow data (song song)
    # ═══════════════════════════════════════════════════════
    from concurrent.futures import ThreadPoolExecutor
    routes_list = list(HANOI_ROUTES.items())

    def get_flow_wrapper(item):
        name, info = item
        return name, fetch_tomtom_flow(info['lat'], info['lon'])

    with ThreadPoolExecutor(max_workers=len(HANOI_ROUTES)) as executor:
        flow_results = dict(executor.map(get_flow_wrapper, routes_list))

    # ═══════════════════════════════════════════════════════
    # Predictions: dùng all_predictions đã tính sẵn từ app.py (có cache)
    # ═══════════════════════════════════════════════════════
    if all_predictions:
        # Sử dụng predictions đã cache từ app.py (không tính lại)
        pass
    else:
        # Fallback: tính lại nếu không có cache (backward compat)
        all_predictions = {}
        for route_name in route_names:
            preds = []
            for h in range(current_hour, min(current_hour + 7, 24)):
                try:
                    speed = predict_speed_for_route(route_name, h)
                    preds.append({'hour': h, 'speed': round(float(speed), 1)})
                except Exception:
                    preds.append({'hour': h, 'speed': 30.0})
            all_predictions[route_name] = preds

    # ═══════════════════════════════════════════════════════
    # SECTION 2: Cards tình trạng 3 tuyến đường
    # ═══════════════════════════════════════════════════════
    st.markdown("### 🚦 Dự Báo Tình Trạng 3 Tuyến Đường")
    cols = st.columns(3)

    for idx, route_name in enumerate(route_names):
        with cols[idx]:
            preds = all_predictions[route_name]
            flow = flow_results.get(route_name)

            current_pred_speed = preds[0]['speed'] if preds else 30.0
            status_text, status_color, status_emoji = get_speed_status(current_pred_speed)

            # Tìm thời điểm tắc đường sắp tới
            congestion_alert = ""
            for p in preds[1:]:
                if p['speed'] < 20:
                    delta_h = p['hour'] - current_hour
                    if delta_h == 1:
                        time_text = "~60 phút nữa"
                    else:
                        time_text = f"~{delta_h} tiếng nữa"
                    congestion_alert = (
                        f"⚠️ Dự báo tắc đường lúc <b>{p['hour']}:00</b>"
                        f"<br>({time_text} — tốc độ giảm còn ~{p['speed']} km/h)"
                    )
                    break

            # Thông tin real-time từ TomTom
            realtime_line = ""
            if flow:
                realtime_line = f"📡 Thực tế: <b>{flow['current_speed']} km/h</b>"

            is_selected = (route_name == selected_route)
            border = f"border: 2px solid {status_color};" if is_selected else f"border-left: 4px solid {status_color};"
            shadow = f"box-shadow: 0 4px 15px {status_color}33;" if is_selected else ""

            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {status_color}18, {status_color}08);
                        {border} padding: 16px; border-radius: 10px; margin-bottom: 8px; {shadow}">
                <div style="font-size: 0.95rem; font-weight: 700; color: #1E3A8A; margin-bottom: 8px;">
                    {route_descs[route_name]}
                </div>
                <div style="font-size: 1.15rem; margin-bottom: 6px;">
                    {status_emoji} <b>{status_text}</b>
                </div>
                <div style="font-size: 0.9rem; margin-bottom: 3px;">
                    🤖 Dự báo AI: <b>{current_pred_speed} km/h</b>
                </div>
                {f'<div style="font-size: 0.85rem; margin-bottom: 3px;">{realtime_line}</div>' if realtime_line else ''}
                <hr style="border: none; border-top: 1px solid {status_color}30; margin: 8px 0;">
                {f'<div style="font-size: 0.85rem; color: #DC2626; font-weight: 600;">{congestion_alert}</div>' if congestion_alert else '<div style="font-size: 0.85rem; color: #059669; font-weight: 600;">✅ Không có dự báo tắc trong 6h tới</div>'}
            </div>
            """, unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════
    # SECTION 3 & 4: Bản Đồ + Biểu đồ chi tiết
    # @st.fragment: khi user đổi tuyến selectbox trong tab này,
    # chỉ render lại bản đồ + biểu đồ — không re-run app.py
    # ═══════════════════════════════════════════════════════
    @st.fragment
    def _render_map_and_chart(selected_route_frag):
        st.markdown("### 🗺️ Bản Đồ Tuyến Đường")

        # --- Build markers JS ---
        markers_js = ""
        for route_name in route_names:
            route_info = HANOI_ROUTES[route_name]
            lat = route_info['lat']
            lon = route_info['lon']
            flow = flow_results.get(route_name)
            preds = all_predictions.get(route_name, [])
            pred_speed = preds[0]['speed'] if preds else 30.0
            status_text, status_color, _ = get_speed_status(pred_speed)

            # Congestion forecast line
            congestion_line = "✅ Ổn trong 6h tới"
            congestion_line_color = "#059669"
            for p in preds[1:]:
                if p['speed'] < 20:
                    delta_h = p['hour'] - current_hour
                    congestion_line = f"⚠️ Dự báo tắc lúc {p['hour']}:00 (~{delta_h}h nữa)"
                    congestion_line_color = "#DC2626"
                    break

            # Popup HTML
            popup_parts = [
                f"<div style=\\'min-width:220px;font-family:system-ui,sans-serif;\\'>" ,
                f"<h3 style=\\'margin:0 0 8px 0;color:#1E3A8A;font-size:14px;\\'>{ route_info['desc']}</h3>",
                f"<div style=\\'margin:4px 0;font-size:13px;\\'><span style=\\'color:{status_color};font-weight:bold;\\'>{status_text}</span></div>",
                f"<div style=\\'margin:3px 0;font-size:12px;\\'>\U0001f916 Dự báo: <b>{pred_speed} km/h</b></div>",
            ]
            if flow:
                popup_parts.append(
                    f"<div style=\\'margin:3px 0;font-size:12px;\\'>\U0001f4e1 Thực tế: <b>{flow['current_speed']} km/h</b></div>"
                )
            popup_parts.extend([
                f"<hr style=\\'margin:8px 0;border:none;border-top:1px solid #e5e7eb;\\'>" ,
                f"<div style=\\'margin:3px 0;font-size:11px;color:{congestion_line_color};font-weight:bold;\\'>{congestion_line}</div>",
                f"</div>",
            ])
            popup_html = "".join(popup_parts)

            is_selected_route = (route_name == selected_route_frag)
            radius = 14 if is_selected_route else 10
            border_color = "#1E3A8A" if is_selected_route else "#fff"
            weight = 4 if is_selected_route else 3
            open_popup = ".openPopup()" if is_selected_route else ""

            markers_js += f"""
            L.circleMarker([{lat}, {lon}], {{
                radius: {radius},
                fillColor: '{status_color}',
                color: '{border_color}',
                weight: {weight},
                opacity: 1,
                fillOpacity: 0.9
            }}).addTo(map).bindPopup('{popup_html}'){open_popup};
            """

        # --- Build polyline JS ---
        polylines_js = ""
        for route_name, coords in ROUTE_POLYLINES.items():
            preds = all_predictions.get(route_name, [])
            pred_speed = preds[0]['speed'] if preds else 30.0
            _, line_color, _ = get_speed_status(pred_speed)

            is_selected_route = (route_name == selected_route_frag)
            weight = 7 if is_selected_route else 3
            opacity = 1.0 if is_selected_route else 0.45
            dash_array = "" if is_selected_route else "dashArray: '8, 6',"
            route_color = line_color if is_selected_route else "#6366F1"
            coords_json = json.dumps(coords)

            polylines_js += f"""
            L.polyline({coords_json}, {{
                color: '{route_color}',
                weight: {weight},
                opacity: {opacity},
                {dash_array}
                lineCap: 'round',
                lineJoin: 'round'
            }}).addTo(map);
            """

        # --- Traffic tile layers ---
        traffic_layers_js = ""
        layer_control_js = ""
        if TOMTOM_API_KEY:
            traffic_layers_js = f"""
            var trafficFlow = L.tileLayer(
                'https://api.tomtom.com/traffic/map/4/tile/flow/relative0/{{z}}/{{x}}/{{y}}.png?key={TOMTOM_API_KEY}&thickness=10',
                {{ maxZoom: 18, opacity: 0.65 }}
            ).addTo(map);
            var trafficIncidents = L.tileLayer(
                'https://api.tomtom.com/traffic/map/4/tile/incidents/s3/{{z}}/{{x}}/{{y}}.png?key={TOMTOM_API_KEY}',
                {{ maxZoom: 18, opacity: 0.75 }}
            );
            """
            layer_control_js = """
            L.control.layers(null, {
                "🚗 Traffic Flow": trafficFlow,
                "⚠️ Sự cố giao thông": trafficIncidents
            }, {collapsed: false, position: 'bottomleft'}).addTo(map);
            """

        # Center on selected route
        center_lat = HANOI_ROUTES[selected_route_frag]['lat']
        center_lon = HANOI_ROUTES[selected_route_frag]['lon']

        # --- Legend HTML ---
        legend_html = """
        var legend = L.control({position: 'topright'});
        legend.onAdd = function(map) {
            var div = L.DomUtil.create('div', 'info legend');
            div.style.cssText = 'background:white;padding:10px 14px;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.15);font-family:system-ui,sans-serif;font-size:12px;line-height:1.8;';
            div.innerHTML =
                '<b style="font-size:13px;">Chú thích</b><br>' +
                '<span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#10B981;margin-right:6px;vertical-align:middle;"></span> ≥30 km/h Thông thoáng<br>' +
                '<span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#F59E0B;margin-right:6px;vertical-align:middle;"></span> 20-30 km/h Ùn ứ<br>' +
                '<span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#EF4444;margin-right:6px;vertical-align:middle;"></span> <20 km/h Tắc nghữn';
            return div;
        };
        legend.addTo(map);
        """

        map_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
            <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
            <style>
                body {{ margin: 0; padding: 0; }}
                #map {{ width: 100%; height: 520px; border-radius: 12px; }}
                .leaflet-popup-content-wrapper {{
                    border-radius: 10px;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                }}
            </style>
        </head>
        <body>
            <div id="map"></div>
            <script>
                var map = L.map('map', {{
                    center: [{center_lat}, {center_lon}],
                    zoom: 14,
                    zoomControl: true
                }});

                // Base map: OpenStreetMap
                L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                    maxZoom: 19,
                    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
                }}).addTo(map);

                // Traffic layers (TomTom)
                {traffic_layers_js}
                {layer_control_js}

                // Polylines cho tuyến đường
                {polylines_js}

                // Markers
                {markers_js}

                // Legend
                {legend_html}
            </script>
        </body>
        </html>
        """

        st.components.v1.html(map_html, height=540)

        # ═══════════════════════════════════════════════════════
        # SECTION 4: Biểu đồ dự báo tốc độ theo giờ
        # ═══════════════════════════════════════════════════════
        st.markdown(f"### 📊 Dự Báo Chi Tiết — {route_descs[selected_route_frag]}")
        st.caption(f"Mô hình: {selected_model_type} · Ngày: {selected_date} · Từ {current_hour}:00 → {min(current_hour + 6, 23)}:00")

        preds = all_predictions[selected_route_frag]
        if preds:
            try:
                import plotly.graph_objects as go

                hours = [f"{p['hour']}:00" for p in preds]
                speeds = [p['speed'] for p in preds]
                colors = []
                for s in speeds:
                    _, c, _ = get_speed_status(s)
                    colors.append(c)

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=hours, y=speeds,
                    marker_color=colors,
                    text=[f"{s} km/h" for s in speeds],
                    textposition='outside',
                    textfont=dict(size=12, color='#1E3A8A'),
                    hovertemplate='%{x}: <b>%{y} km/h</b><extra></extra>'
                ))

                # Ngưỡng tắc nghữn
                fig.add_hline(y=20, line_dash="dash", line_color="#EF4444",
                              annotation_text="Ngưỡng tắc nghữn (20 km/h)",
                              annotation_position="top right",
                              annotation_font_size=10,
                              annotation_font_color="#EF4444")
                fig.add_hline(y=30, line_dash="dash", line_color="#F59E0B",
                              annotation_text="Ngưỡng ùn ứ (30 km/h)",
                              annotation_position="top right",
                              annotation_font_size=10,
                              annotation_font_color="#F59E0B")

                fig.update_layout(
                    xaxis_title="Giờ",
                    yaxis_title="Tốc độ dự báo (km/h)",
                    height=340,
                    margin=dict(t=30, b=50, l=50, r=80),
                    yaxis=dict(range=[0, max(speeds) + 15]),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    showlegend=False,
                )
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.06)')
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.06)')

                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                # Fallback nếu không có plotly
                import pandas as pd
                df_chart = pd.DataFrame({
                    'Giờ': [f"{p['hour']}:00" for p in preds],
                    'Tốc độ (km/h)': [p['speed'] for p in preds]
                }).set_index('Giờ')
                st.bar_chart(df_chart)

        st.caption("📍 Bản đồ nền: OpenStreetMap · 🚗 Giao thông: TomTom API · 🤖 Dự báo: AI Model")

    # Gọi fragment với tuyến đường hiện tại
    _render_map_and_chart(selected_route)
