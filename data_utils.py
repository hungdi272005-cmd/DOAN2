import pandas as pd
import numpy as np
import os
import joblib
import sys
import io

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


# Paths
TOMTOM_PATH = 'data_tomtom.csv'
REALTIME_PATH = 'hanoi_traffic_realtime.csv'
PROFILES_PATH = 'tomtom_profiles.pkl'

# Target feature columns
TARGET_TOMTOM_FEATURES = ['TrafficIndexLive', 'JamsCount', 'JamsLengthInKms', 'JamsDelay']

def clean_and_load_tomtom():
    """Load and clean data_tomtom.csv."""
    if not os.path.exists(TOMTOM_PATH):
        raise FileNotFoundError(f"❌ File {TOMTOM_PATH} không tồn tại!")
    
    df = pd.read_csv(TOMTOM_PATH)
    # Strip spaces and double quotes from columns
    df.columns = df.columns.str.strip().str.replace('"', '')
    
    # Parse LocalDateTime
    df['LocalDateTime'] = pd.to_datetime(df['LocalDateTime'])
    
    # Extract time features
    df['day_of_week'] = df['LocalDateTime'].dt.dayofweek
    df['hour'] = df['LocalDateTime'].dt.hour
    
    return df

def generate_profiles():
    """Create and save historical traffic profiles."""
    print("🔄 Đang tạo profile lưu lượng giao thông lịch sử từ data_tomtom.csv...")
    df_tomtom = clean_and_load_tomtom()
    
    # Group by day of week and hour to compute means
    profiles = df_tomtom.groupby(['day_of_week', 'hour'])[TARGET_TOMTOM_FEATURES].mean().reset_index()
    
    # Save using joblib
    joblib.dump(profiles, PROFILES_PATH)
    print(f"✅ Đã tạo và lưu profile vào {PROFILES_PATH}")
    return profiles

def load_profiles():
    """Load the historical profiles. Create them if not exists."""
    if not os.path.exists(PROFILES_PATH):
        return generate_profiles()
    return joblib.load(PROFILES_PATH)

def load_and_merge_data(data_source='realtime'):
    """
    Load data (realtime) and merge with TomTom city-wide traffic index.
    Fills missing matches using historical profiles.
    """
    if data_source != 'realtime':
        print(f"⚠️ Chú ý: Dữ liệu mock đã bị xóa. Chỉ hỗ trợ source='realtime'. Đang tự động chuyển sang 'realtime'...")
        data_source = 'realtime'
        
    if not os.path.exists(REALTIME_PATH):
        raise FileNotFoundError(f"❌ File {REALTIME_PATH} không tồn tại! Chạy 'python tomtom_collector.py' trước.")
        
    df = pd.read_csv(REALTIME_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Extract basic time features if not present
    if 'hour' not in df.columns:
        df['hour'] = df['timestamp'].dt.hour
    if 'day_of_week' not in df.columns:
        df['day_of_week'] = df['timestamp'].dt.dayofweek
    if 'month' not in df.columns:
        df['month'] = df['timestamp'].dt.month
    if 'is_weekend' not in df.columns:
        df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        
    # Load profile table for fallbacks
    profiles = load_profiles()
    
    # Load raw TomTom index data for actual matches
    df_tomtom = clean_and_load_tomtom()
    
    # Floor timestamps to hour for matching
    df['timestamp_hour'] = df['timestamp'].dt.floor('h')
    df_tomtom['timestamp_hour'] = df_tomtom['LocalDateTime'].dt.floor('h')
    
    # Drop duplicates in tomtom data on timestamp_hour to avoid cartesian product
    df_tomtom_unique = df_tomtom.drop_duplicates(subset=['timestamp_hour'])
    
    # Merge with actual TomTom data
    df_merged = pd.merge(
        df,
        df_tomtom_unique[['timestamp_hour'] + TARGET_TOMTOM_FEATURES],
        on='timestamp_hour',
        how='left'
    )
    
    # Merge with profiles for fallbacks
    # Prefix profile columns with 'hist_'
    profiles_renamed = profiles.rename(columns={col: f'hist_{col}' for col in TARGET_TOMTOM_FEATURES})
    df_merged = pd.merge(
        df_merged,
        profiles_renamed,
        on=['day_of_week', 'hour'],
        how='left'
    )
    
    # Fill NaN values using profile values
    for col in TARGET_TOMTOM_FEATURES:
        df_merged[col] = df_merged[col].fillna(df_merged[f'hist_{col}'])
        # Drop the helper profile column
        df_merged.drop(columns=[f'hist_{col}'], inplace=True)
        
    # Drop other temp columns
    df_merged.drop(columns=['timestamp_hour'], inplace=True, errors='ignore')
    
    print(f"📊 Đã tải và khớp nối dữ liệu: {len(df_merged)} records")
    return df_merged

if __name__ == '__main__':
    # Run profiles generation and print details
    profiles = generate_profiles()
    print(f"Số lượng bản ghi profile (24h x 7 ngày): {len(profiles)}")
    print("\nVí dụ profile thứ 2 lúc 8h sáng:")
    print(profiles[(profiles['day_of_week'] == 0) & (profiles['hour'] == 8)])
    
    # Test load and merge
    try:
        df_test = load_and_merge_data('realtime')
        print("\nCác cột trong dataframe sau khi gộp:")
        print(df_test.columns.tolist())
        print("\nSố lượng giá trị Null trong các cột mới:")
        print(df_test[TARGET_TOMTOM_FEATURES].isnull().sum())
    except Exception as e:
        print(f"⚠️ Lỗi test load/merge: {e}")
