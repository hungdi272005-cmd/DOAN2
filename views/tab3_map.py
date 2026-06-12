import streamlit as st

def render_tab3(TOMTOM_API_KEY, HANOI_ROUTES, fetch_tomtom_flow):
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
                if ratio > 0.7:
                    marker_color = '#10B981'
                elif ratio > 0.4:
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
