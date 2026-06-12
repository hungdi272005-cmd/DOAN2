import streamlit as st
import plotly.express as px
import pandas as pd

def render_tab4(load_realtime_data):
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
            fig_cong.add_hline(y=0.7, line_dash="dash", line_color="#10B981", annotation_text="Thông thoáng")
            fig_cong.add_hline(y=0.4, line_dash="dash", line_color="#F59E0B", annotation_text="Ùn ứ")
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
