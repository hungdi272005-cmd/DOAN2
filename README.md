# 🚦 Hệ Thống Dự Báo Giao Thông Hà Nội (Traffic Forecasting)

Dự án dự báo tốc độ giao thông, mức độ tắc nghẽn trên các tuyến đường chính tại Hà Nội bằng cách ứng dụng mô hình học máy **Random Forest**, **LSTM** và **GRU Deep Learning** kết hợp dữ liệu **thật** từ **TomTom Traffic API** và **OpenWeatherMap API**.

## 🌟 Tính Năng

- **📊 Dự báo AI**: Dự đoán tốc độ giao thông 24h bằng 3 model (RF, LSTM, GRU)
- **🔴 Real-time**: Dữ liệu tốc độ thật từ TomTom Traffic Flow API
- **🗺️ Bản đồ Traffic**: Bản đồ TomTom với lớp phủ giao thông real-time
- **📈 Lịch sử**: Biểu đồ phân tích dữ liệu thu thập theo thời gian
- **🌤️ Thời tiết**: Tích hợp OpenWeatherMap API cho dữ liệu thời tiết thật

## 📂 Cấu Trúc Mã Nguồn

| File | Mô tả |
|------|--------|
| `generate_hanoi_data.py` | Tạo dữ liệu mock lịch sử → `trafficstats_hanoi_mock.csv` |
| `tomtom_collector.py` | **[MỚI]** Thu thập dữ liệu THẬT từ TomTom + OpenWeather → `hanoi_traffic_realtime.csv` |
| `traffic_pipeline.py` | Huấn luyện mô hình Random Forest |
| `train_lstm.py` | Huấn luyện mô hình LSTM Deep Learning |
| `train_gru.py` | Huấn luyện mô hình GRU Deep Learning |
| `app.py` | Giao diện Web (Streamlit) với 4 tabs |

---

## ⚙️ Hướng Dẫn Cài Đặt

### Bước 1: Cài đặt thư viện
```bash
pip install pandas numpy scikit-learn streamlit plotly tensorflow python-dotenv requests
```

### Bước 2: Cấu hình API Keys
Tạo file `.env` trong thư mục gốc:
```
TOMTOM_API_KEY=your_tomtom_api_key_here
OPENWEATHER_API_KEY=your_openweather_api_key_here
```

### Bước 3: Thu thập dữ liệu

**Cách 1: Dùng dữ liệu mock (offline / demo)**
```bash
python generate_hanoi_data.py
```

**Cách 2: Thu thập dữ liệu thật từ TomTom (khuyến nghị)**
```bash
# Test 1 lần
python tomtom_collector.py

# Chạy liên tục mỗi 15 phút (để tích lũy data)
python tomtom_collector.py --schedule

# Thu thập 24h liên tục
python tomtom_collector.py --backfill

# Thu thập liên tục mỗi 5 phút
python tomtom_collector.py --schedule --interval 5

```

### Bước 4: Huấn luyện Mô hình

**Dùng mock data (mặc định):**
```bash
python traffic_pipeline.py
python train_lstm.py
python train_gru.py
```

**Dùng dữ liệu thật TomTom:**
```bash
python traffic_pipeline.py --realtime
python train_lstm.py --realtime
python train_gru.py --realtime
```

**Kết hợp cả mock + thật:**
```bash
python traffic_pipeline.py --combined
python train_lstm.py --combined
python train_gru.py --combined
```

### Bước 5: Khởi động Web App
```bash
streamlit run app.py
```

---

## 🔌 API Keys

| API | Dùng cho | Đăng ký |
|-----|----------|---------|
| TomTom Traffic | Tốc độ real-time, bản đồ, sự cố | [developer.tomtom.com](https://developer.tomtom.com) |
| OpenWeatherMap | Thời tiết thật Hà Nội | [openweathermap.org](https://openweathermap.org/api) |

## 📡 Tuyến Đường Theo Dõi

| Tuyến | Tọa độ | Khu vực |
|-------|--------|---------|
| Nguyễn Trãi | 21.0024, 105.7979 | Thanh Xuân |
| Vành Đai 3 | 20.9952, 105.7872 | Linh Đàm |
| Tôn Đức Thắng | 21.0256, 105.8365 | Đống Đa |
