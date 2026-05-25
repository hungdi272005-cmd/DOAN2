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

# Thu thập liên tục mỗi 5 phút`
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

---

## 🧠 Nguyên Lý Đánh Giá & Phân Loại Ùn Tắc Giao Thông

Hệ thống ứng dụng mô hình học máy hồi quy để dự báo tốc độ giao thông ($v_{pred}$ tính bằng km/h), sau đó quy đổi thành các trạng thái ùn tắc dựa trên **tỷ lệ suy giảm tốc độ** so với **tốc độ chuẩn (Tốc độ tự do - Free-Flow Speed)** của từng tuyến đường.

### 1. Công Thức Tính Toán Cốt Lõi

Tỷ lệ ùn tắc ($R$) được xác định bằng công thức:
$$R = \frac{v_{actual}}{v_{free\_flow}}$$

Trong đó:
* $v_{actual}$: Tốc độ di chuyển thực tế (hoặc tốc độ dự báo của mô hình AI) (km/h).
* $v_{free\_flow}$: Tốc độ tự do lý thuyết (TomTom API) của tuyến đường đó khi hoàn toàn thông thoáng (km/h).

#### Mốc tốc độ chuẩn ($v_{free\_flow}$) động đối với dữ liệu thực tế TomTom:
* **Đường Nguyễn Trãi:** $31 \text{ km/h}$
* **Đường Vành Đai 3:** $41 \text{ km/h}$
* **Đường Tôn Đức Thắng:** $28 \text{ km/h}$

*(Nếu dùng dữ liệu mô phỏng Offline, tốc độ chuẩn lý thuyết được quy ước lần lượt là $40$, $60$ và $35 \text{ km/h}$).*

---

### 2. Các Bài Toán Đánh Giá

#### A. Bài toán Phân lớp Nhị phân (Có Ùn Tắc vs Không Ùn Tắc)
Đây là bài toán thực tế phục vụ trực tiếp cho việc đưa ra cảnh báo giao thông nhanh:
* 🔴 **CÓ ÙN TẮC (Class 1):** Khi **$R \le 80\%$** (Tốc độ giảm từ $20\%$ trở lên so với ngày thường lúc thông thoáng).
* 🟢 **KHÔNG ÙN TẮC (Class 0):** Khi **$R > 80\%$** (Xe cộ lưu thông bình thường hoặc sát nút tốc độ tự do tối đa).

#### B. Bài toán Phân lớp Đa lớp (Chi Tiết Trạng Thái)
Để phân tích chuyên sâu cho các biểu đồ phân tích và điều tiết luồng, hệ thống chia nhỏ thành 3 trạng thái:
1. 🟢 **THÔNG THOÁNG:** Khi **$R > 80\%$** (Không có hiện tượng cản trở, xe đi đúng tốc độ cho phép).
2. 🟡 **ÙN Ứ / CHẬM:** Khi **$40\% < R \le 80\%$** (Mật độ xe đông, tốc độ di chuyển chậm, bắt đầu có hiện tượng dồn ứ).
3. 🔴 **TẮC NGHẼN NẶNG:** Khi **$R \le 40\%$** (Xe cộ ùn tắc nghiêm trọng, tốc độ di chuyển rất thấp, trễ giờ nghiêm trọng).

---

### 🎯 3. Các Chỉ Số Đánh Giá Học Thuật Được Tính Toán
Dựa trên nhãn phân lớp thực tế ($y$) và nhãn dự đoán từ mô hình học máy ($\hat{y}$), hệ thống tự động tính toán các chỉ số sau để báo cáo kết quả đồ án (được hiển thị trực quan tại Tab 5):

* **Độ chính xác (Accuracy):**
  $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
* **Độ chuẩn xác (Precision):**
  $$\text{Precision} = \frac{TP}{TP + FP}$$
* **Độ nhạy / Độ thu hồi (Recall / Ricon):**
  $$\text{Recall} = \frac{TP}{TP + FN}$$
* **F1-Score (Responscore):**
  $$\text{F1-score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

