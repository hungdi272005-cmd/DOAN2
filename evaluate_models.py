"""
evaluate_models.py - Đánh giá độ chính xác mô hình
Tính toán Accuracy, Precision, Recall, F1-score cho cả 3 model:
  Random Forest, LSTM, GRU

Phương pháp: Chuyển bài toán hồi quy (speed_kmh) thành phân lớp
trạng thái tắc nghẽn (Thông thoáng / Ùn ứ / Tắc nghẽn) rồi đánh giá.

Sử dụng:
  python evaluate_models.py              # Dùng mock data
  python evaluate_models.py --realtime   # Dùng dữ liệu thật TomTom
  python evaluate_models.py --combined   # Kết hợp cả mock + thật
"""

import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except:
    pass

import pandas as pd
import numpy as np
import joblib
import json
import os
import argparse
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# ========================================
# Tốc độ chuẩn (free-flow) cho từng tuyến
# ========================================
ROUTE_BASE_SPEED = {
    'Nguyen Trai': 40,
    'Vanh Dai 3': 60,
    'Ton Duc Thang': 35,
}
DEFAULT_BASE_SPEED = 40

# Nhãn phân lớp
CONGESTION_LABELS = {
    0: 'Thông thoáng',
    1: 'Ùn ứ / Chậm',
    2: 'Tắc nghẽn',
}


def speed_to_congestion_class(speed, base_speed):
    """
    Phân lớp trạng thái tắc nghẽn dựa trên tỷ lệ tốc độ / tốc độ chuẩn.
      - 0: Thông thoáng  (ratio > 0.8)
      - 1: Ùn ứ / Chậm   (0.4 < ratio <= 0.8)
      - 2: Tắc nghẽn      (ratio <= 0.4)
    """
    ratio = speed / base_speed if base_speed > 0 else 1
    if ratio > 0.8:
        return 0
    elif ratio > 0.4:
        return 1
    else:
        return 2


def load_data(data_source='mock'):
    """Load dữ liệu theo nguồn."""
    if data_source == 'realtime':
        filepath = 'hanoi_traffic_realtime.csv'
        if not os.path.exists(filepath):
            print("[Error] Chua co file du lieu that. Chay 'python tomtom_collector.py' truoc!")
            return None
        print(f"[Info] Dang dung du lieu THAT tu TomTom: {filepath}")
        df = pd.read_csv(filepath)
    elif data_source == 'combined':
        dfs = []
        if os.path.exists('trafficstats_hanoi_mock.csv'):
            df_mock = pd.read_csv('trafficstats_hanoi_mock.csv')
            df_mock['data_source'] = 'mock'
            dfs.append(df_mock)
            print(f"   Mock data: {len(df_mock)} records")
        if os.path.exists('hanoi_traffic_realtime.csv'):
            df_real = pd.read_csv('hanoi_traffic_realtime.csv')
            dfs.append(df_real)
            print(f"   Real data: {len(df_real)} records")
        if not dfs:
            print("[Error] Khong tim thay file du lieu nao!")
            return None
        df = pd.concat(dfs, ignore_index=True)
        print(f"[Info] Ket hop: Tong {len(df)} records")
    else:
        filepath = 'trafficstats_hanoi_mock.csv'
        print(f"[Info] Dang dung du lieu MOCK: {filepath}")
        df = pd.read_csv(filepath)

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def prepare_features(df):
    """Chuẩn bị features cho dữ liệu."""
    if 'hour' not in df.columns:
        df['hour'] = df['timestamp'].dt.hour
    if 'day_of_week' not in df.columns:
        df['day_of_week'] = df['timestamp'].dt.dayofweek
    if 'month' not in df.columns:
        df['month'] = df['timestamp'].dt.month

    le_route = joblib.load('le_route.pkl')
    le_weather = joblib.load('le_weather.pkl')

    df['route_encoded'] = le_route.transform(df['route'])
    df['weather_encoded'] = le_weather.transform(df['weather'])

    return df


def get_base_speed_for_rows(df):
    """Trả về mảng tốc độ chuẩn tương ứng với từng hàng dữ liệu."""
    if 'free_flow_speed' in df.columns:
        # Sử dụng tốc độ tự do thực tế từ TomTom API làm mốc chuẩn
        return df['free_flow_speed'].values
    return df['route'].map(ROUTE_BASE_SPEED).fillna(DEFAULT_BASE_SPEED).values


def evaluate_single_model(model_name, y_true, y_pred, base_speeds):
    """
    Đánh giá một mô hình với cả chỉ số hồi quy và phân lớp.
    """
    # --- Chỉ số hồi quy ---
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    # --- Phân lớp trạng thái tắc nghẽn ---
    y_true_class = np.array([
        speed_to_congestion_class(s, b) for s, b in zip(y_true, base_speeds)
    ])
    y_pred_class = np.array([
        speed_to_congestion_class(s, b) for s, b in zip(y_pred, base_speeds)
    ])

    # Chỉ số phân lớp
    accuracy = float(accuracy_score(y_true_class, y_pred_class))
    precision = float(precision_score(y_true_class, y_pred_class, average='weighted', zero_division=0))
    recall = float(recall_score(y_true_class, y_pred_class, average='weighted', zero_division=0))
    f1 = float(f1_score(y_true_class, y_pred_class, average='weighted', zero_division=0))

    # Ma trận nhầm lẫn
    labels = sorted(list(set(y_true_class.tolist() + y_pred_class.tolist())))
    cm = confusion_matrix(y_true_class, y_pred_class, labels=[0, 1, 2])

    # Classification report chi tiết cho từng lớp
    report = classification_report(
        y_true_class, y_pred_class,
        labels=[0, 1, 2],
        target_names=['Thông thoáng', 'Ùn ứ / Chậm', 'Tắc nghẽn'],
        output_dict=True,
        zero_division=0
    )

    # Chuyển report thành serializable
    per_class = {}
    for cls_name in ['Thông thoáng', 'Ùn ứ / Chậm', 'Tắc nghẽn']:
        if cls_name in report:
            per_class[cls_name] = {
                'precision': round(report[cls_name]['precision'], 4),
                'recall': round(report[cls_name]['recall'], 4),
                'f1-score': round(report[cls_name]['f1-score'], 4),
                'support': int(report[cls_name]['support']),
            }

    # --- Phân lớp nhị phân (Có ùn tắc / Không ùn tắc) ---
    # Có ùn tắc = 1 (tỷ lệ tốc độ / tốc độ chuẩn <= 0.8)
    # Không ùn tắc = 0 (tỷ lệ tốc độ / tốc độ chuẩn > 0.8)
    y_true_bin = np.array([1 if s / b <= 0.8 else 0 for s, b in zip(y_true, base_speeds)])
    y_pred_bin = np.array([1 if s / b <= 0.8 else 0 for s, b in zip(y_pred, base_speeds)])

    bin_accuracy = float(accuracy_score(y_true_bin, y_pred_bin))
    bin_precision = float(precision_score(y_true_bin, y_pred_bin, average='binary', pos_label=1, zero_division=0))
    bin_recall = float(recall_score(y_true_bin, y_pred_bin, average='binary', pos_label=1, zero_division=0))
    bin_f1 = float(f1_score(y_true_bin, y_pred_bin, average='binary', pos_label=1, zero_division=0))
    bin_cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])

    result = {
        'model_name': model_name,
        'regression': {
            'rmse': round(rmse, 4),
            'mae': round(mae, 4),
            'r2': round(r2, 4),
        },
        'classification': {
            'accuracy': round(accuracy, 4),
            'precision_weighted': round(precision, 4),
            'recall_weighted': round(recall, 4),
            'f1_weighted': round(f1, 4),
        },
        'binary_classification': {
            'accuracy': round(bin_accuracy, 4),
            'precision': round(bin_precision, 4),
            'recall': round(bin_recall, 4),
            'f1_score': round(bin_f1, 4),
            'confusion_matrix': bin_cm.tolist(),
        },
        'confusion_matrix': cm.tolist(),
        'per_class_report': per_class,
        'test_samples': int(len(y_true)),
    }

    return result


def evaluate_all_models(data_source='mock'):
    """Đánh giá tất cả các mô hình có sẵn."""
    print("=" * 60)
    print("  DANH GIA DO CHINH XAC MO HINH")
    print("=" * 60)

    df = load_data(data_source)
    if df is None:
        return None

    df = prepare_features(df)

    features = ['route_encoded', 'weather_encoded', 'is_weekend', 'hour', 'day_of_week', 'month']
    X = df[features].values
    y = df['speed_kmh'].values
    base_speeds = get_base_speed_for_rows(df)

    # Chia tập test giống lúc train (chronological 80/20)
    split_idx = int(len(X) * 0.8)
    X_test = X[split_idx:]
    y_test = y[split_idx:]
    base_speeds_test = base_speeds[split_idx:]
    routes_test = df['route'].iloc[split_idx:].values

    print(f"\n[Info] Tong du lieu: {len(X)} samples")
    print(f"[Info] Tap test: {len(X_test)} samples")

    results = {}

    # ==========================
    # 1. Random Forest
    # ==========================
    print("\n" + "-" * 40)
    print("🌲 Evaluation Random Forest...")
    try:
        rf_model = joblib.load('rf_traffic_model.pkl')
        y_pred_rf = rf_model.predict(X_test)
        results['Random Forest'] = evaluate_single_model(
            'Random Forest', y_test, y_pred_rf, base_speeds_test
        )
        print(f"   [Success] Accuracy: {results['Random Forest']['classification']['accuracy']:.2%}")
        print(f"   [Success] Precision: {results['Random Forest']['classification']['precision_weighted']:.2%}")
        print(f"   [Success] Recall: {results['Random Forest']['classification']['recall_weighted']:.2%}")
        print(f"   [Success] F1-score: {results['Random Forest']['classification']['f1_weighted']:.2%}")
        print(f"   [Success] RMSE: {results['Random Forest']['regression']['rmse']:.2f} km/h")
        print(f"   [Success] MAE: {results['Random Forest']['regression']['mae']:.2f} km/h")
        print(f"   [Success] R2: {results['Random Forest']['regression']['r2']:.4f}")
    except Exception as e:
        print(f"   [Error]: {e}")

    # ==========================
    # 2. LSTM
    # ==========================
    print("\n" + "-" * 40)
    print("🧠 Evaluation LSTM...")
    try:
        lstm_path_keras = 'lstm_traffic_model.keras'
        lstm_path_h5 = 'lstm_traffic_model.h5'
        if os.path.exists(lstm_path_keras) or os.path.exists(lstm_path_h5):
            from tensorflow.keras.models import load_model
            if os.path.exists(lstm_path_keras):
                lstm_model = load_model(lstm_path_keras)
            else:
                lstm_model = load_model(lstm_path_h5)

            scaler_lstm = joblib.load('scaler_X_lstm.pkl')
            X_test_scaled = scaler_lstm.transform(X_test)
            X_test_3d = np.reshape(X_test_scaled, (X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
            y_pred_lstm = lstm_model.predict(X_test_3d, verbose=0).flatten()

            results['LSTM'] = evaluate_single_model(
                'LSTM', y_test, y_pred_lstm, base_speeds_test
            )
            print(f"   [Success] Accuracy: {results['LSTM']['classification']['accuracy']:.2%}")
            print(f"   [Success] Precision: {results['LSTM']['classification']['precision_weighted']:.2%}")
            print(f"   [Success] Recall: {results['LSTM']['classification']['recall_weighted']:.2%}")
            print(f"   [Success] F1-score: {results['LSTM']['classification']['f1_weighted']:.2%}")
            print(f"   [Success] RMSE: {results['LSTM']['regression']['rmse']:.2f} km/h")
            print(f"   [Success] MAE: {results['LSTM']['regression']['mae']:.2f} km/h")
            print(f"   [Success] R2: {results['LSTM']['regression']['r2']:.4f}")
        else:
            print("   [Warning] Khong tim thay model LSTM. Bo qua.")
    except Exception as e:
        print(f"   [Error]: {e}")

    # ==========================
    # 3. GRU
    # ==========================
    print("\n" + "-" * 40)
    print("⚡ Evaluation GRU...")
    try:
        gru_path_keras = 'gru_traffic_model.keras'
        if os.path.exists(gru_path_keras):
            from tensorflow.keras.models import load_model
            gru_model = load_model(gru_path_keras)

            scaler_gru = joblib.load('scaler_X_gru.pkl')
            X_test_scaled = scaler_gru.transform(X_test)
            X_test_3d = np.reshape(X_test_scaled, (X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
            y_pred_gru = gru_model.predict(X_test_3d, verbose=0).flatten()

            results['GRU'] = evaluate_single_model(
                'GRU', y_test, y_pred_gru, base_speeds_test
            )
            print(f"   [Success] Accuracy: {results['GRU']['classification']['accuracy']:.2%}")
            print(f"   [Success] Precision: {results['GRU']['classification']['precision_weighted']:.2%}")
            print(f"   [Success] Recall: {results['GRU']['classification']['recall_weighted']:.2%}")
            print(f"   [Success] F1-score: {results['GRU']['classification']['f1_weighted']:.2%}")
            print(f"   [Success] RMSE: {results['GRU']['regression']['rmse']:.2f} km/h")
            print(f"   [Success] MAE: {results['GRU']['regression']['mae']:.2f} km/h")
            print(f"   [Success] R2: {results['GRU']['regression']['r2']:.4f}")
        else:
            print("   [Warning] Khong tim thay model GRU. Bo qua.")
    except Exception as e:
        print(f"   [Error]: {e}")

    # ==========================
    # Lưu kết quả
    # ==========================
    if results:
        output = {
            'evaluated_at': pd.Timestamp.now().isoformat(),
            'data_source': data_source,
            'total_test_samples': int(len(y_test)),
            'congestion_thresholds': {
                'thong_thoang': 'ratio > 0.8',
                'un_u': '0.4 < ratio <= 0.8',
                'tac_nghen': 'ratio <= 0.4',
            },
            'route_base_speeds': ROUTE_BASE_SPEED,
            'models': results,
        }

        output_path = 'evaluation_results.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"\n{'=' * 60}")
        print(f"[Success] Da luu ket qua danh gia vao: {output_path}")
        print(f"   Tong mo hinh danh gia: {len(results)}")
        print(f"{'=' * 60}")

        return output
    else:
        print("\n[Error] Khong co mo hinh nao duoc danh gia thanh cong.")
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Đánh giá mô hình - Accuracy, Precision, Recall, F1')
    parser.add_argument('--mock', action='store_true', help='Dùng dữ liệu mock')
    parser.add_argument('--combined', action='store_true', help='Kết hợp mock + thật')
    args = parser.parse_args()

    if args.mock:
        data_source = 'mock'
    elif args.combined:
        data_source = 'combined'
    else:
        data_source = 'realtime'

    evaluate_all_models(data_source)
