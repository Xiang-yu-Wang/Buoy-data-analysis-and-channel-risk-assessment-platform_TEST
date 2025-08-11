import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import json
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention
from keras.callbacks import EarlyStopping
import glob

from utils.helpers import initialize_session_state 

# 設置 TensorFlow 日誌級別，抑制 INFO 訊息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# --- Streamlit 頁面設定 (必須是第一個 Streamlit 命令) ---
st.set_page_config(
    page_title="Transformer 模型預測",
    page_icon="🤖",
    layout="wide"
)
initialize_session_state()

st.title("🤖 海洋數據 Transformer 模型預測")
st.markdown("使用 Transformer 類神經網絡預測海洋數據的未來趨勢。")

# --- 輔助函數和數據載入 ---
CONFIG_PATH = 'config.json'

@st.cache_data
def load_config():
    """載入配置檔"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    possible_config_paths = [
        os.path.join(current_dir, CONFIG_PATH), # 當前頁面所在的目錄
        os.path.join(current_dir, '..', CONFIG_PATH) # 應用根目錄
    ]

    for path in possible_config_paths:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
                # 兼容舊版配置中的經緯度鍵名
                if "STATION_COORDS" in config_data:
                    new_coords = {}
                    for station, coords in config_data["STATION_COORDS"].items():
                        new_coords[station] = {
                            "latitude": coords.get("lat", coords.get("latitude")),
                            "longitude": coords.get("lon", coords.get("longitude"))
                        }
                    config_data["STATION_COORDS"] = new_coords
                return config_data
    st.error(f"錯誤: 配置檔 '{CONFIG_PATH}' 未找到。請確保它存在於應用程式的根目錄或 Streamlit 頁面所在目錄。")
    return {}

config = load_config()

BASE_DATA_PATH_CONFIG = config.get("BASE_DATA_PATH", "資料檔/浮標資料")
# 構建到數據文件夾的絕對路徑
BASE_DATA_PATH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', BASE_DATA_PATH_CONFIG))

STATION_COORDS = config.get("STATION_COORDS", {})
PARAMETER_INFO = config.get("PARAMETER_INFO", {})
DATA_SUBFOLDERS_PRIORITY = config.get("DATA_SUBFOLDERS_PRIORITY", ["qc", "QC", "real time", "real_time", "RealTime", "Real Time", "realtime"])
CHINESE_FONT_PATH = config.get("CHINESE_FONT_PATH")

@st.cache_data(ttl=3600, show_spinner="正在載入並預處理數據...")
def load_data_for_page(station_name, param_info_map):
    """
    載入指定測站的數據，處理多個子文件夾，並進行初步的時間序列解析。
    優化：增加日期格式推斷，避免 UserWarning。
    """
    st.info(f"嘗試從基本路徑 `{BASE_DATA_PATH}` 載入測站 `{station_name}` 的數據。")
    station_data_base_path = os.path.join(BASE_DATA_PATH, station_name)

    all_dfs = []
    found_any_file = False

    csv_header_row = 0
    if station_name in ["小琉球資料浮標", "南灣資料浮標", "蘇澳資料浮標"]:
        csv_header_row = 1
        st.info(f"檢測到測站 '{station_name}'，將使用 CSV 文件的 **第二行** 作為列名 (header=1)。")
    else:
        st.info(f"未明確指定測站 '{station_name}' 的 CSV 檔頭行，將預設使用 **第一行** 作為列名 (header=0)。")

    for subfolder in DATA_SUBFOLDERS_PRIORITY:
        folder_path = os.path.join(station_data_base_path, subfolder)
        if os.path.isdir(folder_path):
            csv_files = (glob.glob(os.path.join(folder_path, '*.csv')) +
                         glob.glob(os.path.join(folder_path, '*.CSV')))
            if csv_files:
                st.info(f"在 `{folder_path}` 中找到 {len(csv_files)} 個 CSV 檔案。")
                found_any_file = True
                for file_path in sorted(csv_files):
                    try:
                        encodings = ['utf-8', 'latin1', 'big5', 'cp950']
                        df_part = None
                        for enc in encodings:
                            try:
                                df_part = pd.read_csv(file_path, header=csv_header_row, encoding=enc, engine='python')
                                break
                            except UnicodeDecodeError:
                                continue
                        if df_part is None:
                            st.warning(f"文件 '{file_path}' 無法使用常見編碼解析。跳過此文件。")
                            continue

                        time_col = None
                        possible_time_cols = ['Time', 'time', 'UTC', 'GMT', 'Local_Time', 'Date', 'DateTime', 'TIME_UTC', 'Time (UTC)', 'time(UTC)', 'Time (LST)']
                        actual_time_cols_in_df = [col for col in df_part.columns if col in possible_time_cols]
                        
                        # --- 優化日期時間解析邏輯 ---
                        for col in actual_time_cols_in_df:
                            df_part[col] = df_part[col].astype(str).str.strip() # 確保是字串並去除空白
                            
                            # 嘗試多種常見日期時間格式
                            # 注意：您可以根據實際數據中可能出現的格式進行增補或調整
                            possible_formats = [
                                '%Y-%m-%d %H:%M:%S', # 2023-01-01 15:30:00
                                '%Y/%m/%d %H:%M:%S', # 2023/01/01 15:30:00
                                '%Y-%m-%d %H:%M',   # 2023-01-01 15:30
                                '%Y/%m/%d %H:%M',   # 2023/01/01 15:30
                                '%Y-%m-%d %H',      # 2023-01-01 15
                                '%Y/%m/%d %H',      # 2023/01/01 15
                                '%Y-%m-%d',         # 2023-01-01
                                '%Y/%m/%d',         # 2023/01/01
                                '%m/%d/%Y %H:%M:%S', # 01/01/2023 15:30:00 (美式)
                                '%d-%m-%Y %H:%M:%S'  # 01-01-2023 15:30:00 (歐式)
                            ]
                            
                            parsed_dates = pd.Series(dtype='datetime64[ns]') # 初始化為空 Series
                            best_valid_ratio = 0
                            
                            for fmt in possible_formats:
                                temp_parsed = pd.to_datetime(df_part[col], format=fmt, errors='coerce')
                                current_valid_ratio = temp_parsed.count() / len(df_part) if len(df_part) > 0 else 0
                                
                                if current_valid_ratio > best_valid_ratio:
                                    best_valid_ratio = current_valid_ratio
                                    parsed_dates = temp_parsed
                                    if best_valid_ratio == 1.0: # 如果完美解析，就停止嘗試其他格式
                                        break
                            
                            # 如果嘗試了所有格式後，解析成功率仍低於閾值，則回退到自動推斷（可能會有警告）
                            if best_valid_ratio < 0.5 and len(df_part) > 0: # 假設 50% 是可接受的最低成功率
                                st.warning(f"對於文件 '{file_path}' 中的時間列 '{col}'，無法通過常見格式解析，嘗試自動推斷。這可能導致性能問題或部分日期錯誤。")
                                parsed_dates = pd.to_datetime(df_part[col], errors='coerce') # 回退到自動推斷
                            
                            if not parsed_dates.isnull().all() and parsed_dates.count() / len(df_part) > 0.5:
                                time_col = col
                                df_part['ds'] = parsed_dates
                                break

                        if time_col is None or df_part['ds'].isnull().all():
                            st.warning(f"文件 '{file_path}' 中未找到有效的時間列 ({', '.join(possible_time_cols)})，或時間格式無法解析。跳過此文件。")
                            continue
                        
                        df_part.set_index('ds', inplace=True)
                        all_dfs.append(df_part)
                    except Exception as e:
                        st.warning(f"載入或處理文件 '{file_path}' 時發生錯誤：{e}。跳過此文件。")
                        continue
            else:
                st.info(f"在 `{folder_path}` 中沒有找到 CSV 檔案。")

    if not found_any_file:
        st.error(f"錯誤：在測站 '{station_name}' 的任何指定子文件夾中都沒有找到有效的數據文件。")
        st.info(f"預期的測站數據根路徑: `{station_data_base_path}`")
        st.info(f"嘗試尋找的子文件夾: `{', '.join(DATA_SUBFOLDERS_PRIORITY)}`")
        return pd.DataFrame()

    if not all_dfs:
        st.error(f"錯誤：雖然找到了 CSV 檔案，但沒有任何檔案成功載入並解析出有效時間序列數據。")
        return pd.DataFrame()

    combined_df = pd.concat(all_dfs).sort_index()
    combined_df = combined_df[~combined_df.index.duplicated(keep='first')]

    cleaned_df = combined_df.copy()

    final_cols_to_keep = []
    for param_key, param_info in param_info_map.items():
        if param_key in cleaned_df.columns:
            cleaned_df[param_key] = pd.to_numeric(cleaned_df[param_key], errors='coerce')
            valid_ratio = cleaned_df[param_key].count() / len(cleaned_df) if len(cleaned_df) > 0 else 0

            if param_info.get("type") == "linear" and valid_ratio > 0.5:
                final_cols_to_keep.append(param_key)
            elif param_info.get("type") == "circular" and valid_ratio > 0.5: # 允許 circular 數據存在但不納入線性預測
                 pass
            else:
                st.info(f"列 '{param_key}' (顯示名稱: {param_info.get('display_zh', 'N/A')}) 因數據類型不符、空值過多 ({valid_ratio*100:.2f}%) 或非線性類型而被排除在主要分析之外。")
        else:
            st.info(f"配置文件中的參數 '{param_info.get('display_zh', param_key)}' (原始列名: '{param_key}') 未在數據文件中找到。")

    cleaned_df = cleaned_df[final_cols_to_keep]

    if cleaned_df.empty:
        st.error(f"錯誤：合併並清理後的數據為空。請檢查原始文件內容和列名是否與 config.json 匹配。")
        return pd.DataFrame()
    
    cleaned_df.reset_index(inplace=True) 

    return cleaned_df

def analyze_data_quality(df, relevant_params):
    """分析數據品質，提供缺失值、零值、負值和異常值報告。"""
    quality_metrics = {}
    for param_col in relevant_params:
        if param_col in df.columns:
            total_records = len(df)
            missing_count = df[param_col].isnull().sum()
            valid_count = total_records - missing_count
            missing_percentage = (missing_count / total_records) * 100 if total_records > 0 else 0

            if pd.api.types.is_numeric_dtype(df[param_col]):
                zero_count = (df[param_col] == 0).sum()
                negative_count = (df[param_col] < 0).sum()
                outlier_iqr_count = 0

                if valid_count > 0:
                    Q1 = df[param_col].quantile(0.25)
                    Q3 = df[param_col].quantile(0.75)
                    IQR = Q3 - Q1
                    if IQR > 0:
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        outlier_iqr_count = df[(df[param_col] < lower_bound) | (df[param_col] > upper_bound)][param_col].count()

                quality_metrics[param_col] = {
                    'total_records': total_records,
                    'valid_count': valid_count,
                    'missing_count': missing_count,
                    'missing_percentage': missing_percentage,
                    'zero_count': zero_count,
                    'negative_count': negative_count,
                    'outlier_iqr_count': outlier_iqr_count
                }
            else:
                quality_metrics[param_col] = {
                    'total_records': total_records, 'valid_count': valid_count, 'missing_count': missing_count,
                    'missing_percentage': 100.0, 'is_numeric': False
                }
        else:
            quality_metrics[param_col] = {
                'total_records': 0, 'valid_count': 0, 'missing_count': 0,
                'missing_percentage': 100.0, 'is_numeric': False, 'status': '列不存在'
            }
    return quality_metrics


# --- Transformer 模型輔助函數 ---
def build_transformer_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0.2):
    """
    構建一個基於 Transformer 編碼器結構的時間序列預測模型。
    input_shape: (sequence_length, features)
    """
    inputs = Input(shape=input_shape)
    x = inputs

    # Transformer Blocks
    for _ in range(num_transformer_blocks):
        # Attention and Normalization
        x = LayerNormalization(epsilon=1e-6)(x)
        attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=head_size, dropout=dropout)(x, x)
        x = x + attn_output # Residual connection
        x = LayerNormalization(epsilon=1e-6)(x)

        # Feed Forward Network
        ffn_output = Dense(ff_dim, activation="relu")(x)
        ffn_output = Dropout(dropout)(ffn_output)
        ffn_output = Dense(input_shape[-1])(ffn_output) # Output dim matches input feature dim
        x = x + ffn_output # Residual connection

    # Flatten the output for the MLP head
    x = tf.keras.layers.GlobalAveragePooling1D()(x) # 匯總序列信息，將 (batch_size, sequence_length, features) 變成 (batch_size, features)
    
    # MLP Head
    for dim in mlp_units:
        x = Dense(dim, activation="relu")(x)
        x = Dropout(dropout)(x)
    
    outputs = Dense(1)(x) # 輸出單個預測值

    return Model(inputs=inputs, outputs=outputs)

def create_sequences(data, look_back):
    """
    將時間序列數據轉換為模型所需的序列 (X) 和目標 (y)。
    X: 過去 look_back 個時間步的數據
    y: 下一個時間步的數據
    """
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), 0]) # 獲取歷史序列
        y.append(data[i + look_back, 0])     # 獲取目標值
    return np.array(X), np.array(y)


# --- 側邊欄：Transformer 預測設定控制項 ---
st.sidebar.header("Transformer 預測設定")

locations = list(STATION_COORDS.keys())
if not locations:
    st.sidebar.warning("請在 `config.json` 的 `STATION_COORDS` 中配置測站資訊。")
    st.stop()

selected_station = st.sidebar.selectbox("選擇測站:", locations, key='pages_11_transformer_station')

predictable_params_config_map = {
    col_name: info["display_zh"] for col_name, info in PARAMETER_INFO.items()
    if info.get("type") == "linear" # 只選擇線性參數進行預測
}

# 預載入數據以動態獲取可用參數
df_initial_check = load_data_for_page(selected_station, PARAMETER_INFO)

available_predictable_params_display_to_col = {}
for col_name, display_name in predictable_params_config_map.items():
    if col_name in df_initial_check.columns and pd.api.types.is_numeric_dtype(df_initial_check[col_name]):
        if df_initial_check[col_name].count() > 0: # 確保該列有非空數據
            available_predictable_params_display_to_col[display_name] = col_name

if not available_predictable_params_display_to_col:
    st.sidebar.error("載入數據後，沒有可供預測的有效數值型參數。請檢查數據文件和 `config.json` 中的參數配置。")
    st.stop()

selected_param_display = st.sidebar.selectbox("選擇預測參數:", list(available_predictable_params_display_to_col.keys()), key='pages_11_transformer_param_display')
selected_param_col = available_predictable_params_display_to_col[selected_param_display]

param_info_original = PARAMETER_INFO.get(selected_param_col, {})
selected_param_display_original = param_info_original.get("display_zh", selected_param_col)
param_unit = param_info_original.get("unit", "")


st.sidebar.markdown("---")
st.sidebar.subheader("預測時間設定")

prediction_frequencies = {
    "小時 (H)": "h",
    "天 (D)": "D",
    "週 (W)": "W",
    "月 (M)": "M",
    "年 (Y)": "Y"
}
selected_prediction_freq_display = st.sidebar.selectbox(
    "選擇預測頻次:",
    list(prediction_frequencies.keys()),
    key='pages_11_prediction_frequency'
)
selected_freq_pandas = prediction_frequencies[selected_prediction_freq_display]

forecast_period_value = st.sidebar.number_input(
    f"預測未來多久 ({selected_prediction_freq_display.split(' ')[0]}):",
    min_value=1,
    max_value=365 if selected_freq_pandas == 'D' else 8760 if selected_freq_pandas == 'h' else 12, # 調整最大值
    value=24 if selected_freq_pandas == 'h' else 7 if selected_freq_pandas == 'D' else 1,
    step=1,
    key='pages_11_forecast_period_value'
)

# --- 數據訓練時間範圍選擇 ---
st.sidebar.markdown("---")
st.sidebar.subheader("訓練數據時間範圍")

if not df_initial_check.empty and 'ds' in df_initial_check.columns:
    min_date_available = df_initial_check['ds'].min().date()
    max_date_available = df_initial_check['ds'].max().date()
else:
    min_date_available = pd.to_datetime('1990-01-01').date() # 預設起始日期
    max_date_available = pd.Timestamp.now().date() # 預設結束日期
    st.warning("無法從載入的數據中獲取時間範圍。使用預設日期範圍。")

default_start_date = min_date_available
default_end_date = max_date_available

train_start_date = st.sidebar.date_input(
    "訓練數據開始日期:",
    value=default_start_date,
    min_value=min_date_available,
    max_value=max_date_available,
    key='pages_11_train_start_date'
)
train_end_date = st.sidebar.date_input(
    "訓練數據結束日期:",
    value=default_end_date,
    min_value=min_date_available,
    max_value=max_date_available,
    key='pages_11_train_end_date'
)

if train_start_date >= train_end_date:
    st.sidebar.error("訓練數據開始日期必須早於結束日期。")
    st.stop()


# --- 數據預處理選項 ---
st.sidebar.markdown("---")
st.sidebar.subheader("數據預處理")
missing_value_strategy = st.sidebar.selectbox(
    "缺失值處理:",
    options=['前向填充 (ffill)', '後向填充 (bfill)', '線性插值 (interpolate)', '移除缺失值 (dropna)'],
    key='pages_11_missing_strategy'
)

apply_smoothing = st.sidebar.checkbox("應用數據平滑", value=False, key='pages_11_apply_smoothing')
smoothing_window = 1

if apply_smoothing:
    smoothing_window = st.sidebar.slider("平滑處理 (移動平均視窗):", min_value=1, max_value=24, value=3, step=1,
                                         help="移動平均視窗大小（單位與預測頻次相同）。1 表示不進行平滑處理。數值越大，數據越平滑，但可能丟失細節。")


# --- Transformer 模型參數 ---
st.sidebar.markdown("---")
st.sidebar.subheader("Transformer 模型參數")

look_back = st.sidebar.slider("回溯時間步 (look_back):", min_value=1, max_value=48, value=12, step=1,
                              help="Transformer 模型在預測下一個時間點時考慮多少個過去的時間點。")
head_size = st.sidebar.slider("注意力頭維度 (head_size):", min_value=32, max_value=256, value=64, step=32,
                              help="Transformer 注意力頭的維度。建議為 2 的冪次。")
num_heads = st.sidebar.slider("注意力頭數量 (num_heads):", min_value=1, max_value=8, value=4, step=1,
                             help="多頭注意力機制中的頭數量。確保 head_size 能被 num_heads 整除。")
ff_dim = st.sidebar.slider("前饋網絡維度 (ff_dim):", min_value=64, max_value=512, value=128, step=64,
                           help="Transformer 塊中前饋網絡層的維度。通常為 head_size 的 2-4 倍。")
num_transformer_blocks = st.sidebar.slider("Transformer 區塊數量:", min_value=1, max_value=5, value=2, step=1,
                                            help="模型中堆疊的 Transformer 編碼器區塊的數量。")
mlp_units_options = [32, 64, 128, 256]
mlp_units = st.sidebar.multiselect("MLP 層單元數:", options=mlp_units_options, default=[128],
                                   help="預測頭部的多層感知器 (MLP) 層的單元數。可以有多層。")
epochs = st.sidebar.number_input("訓練迭代次數 (Epochs):", min_value=10, max_value=500, value=100, step=10,
                                 help="模型在整個訓練數據集上進行訓練的次數。")
batch_size = st.sidebar.number_input("批次大小 (Batch Size):", min_value=1, max_value=128, value=32, step=8,
                                     help="每次訓練迭代中使用的樣本數量。")
dropout_rate = st.sidebar.slider("Dropout 比率:", min_value=0.0, max_value=0.5, value=0.2, step=0.05,
                                 help="防止過擬合的 Dropout 層比率。")
validation_split = st.sidebar.slider("驗證集比例:", min_value=0.0, max_value=0.5, value=0.1, step=0.05,
                                     help="用於模型訓練期間驗證的數據比例。")
patience = st.sidebar.number_input("早停耐心值 (Patience):", min_value=5, max_value=50, value=10, step=5,
                                  help="如果驗證損失在這麼多個 epochs 內沒有改善，訓練將停止。")

# --- 執行預測按鈕 ---
if st.sidebar.button("🤖 執行 Transformer 預測"):
    # 檢查 TensorFlow 是否可用
    if not tf.test.is_built_with_cuda() and not tf.config.list_physical_devices('GPU'):
        st.warning("警告: TensorFlow 未啟用 GPU 加速。模型訓練可能較慢。")

    df = load_data_for_page(selected_station, PARAMETER_INFO)

    if df.empty or selected_param_col not in df.columns:
        if df.empty:
            st.error(f"所選測站 '{selected_station}' 沒有成功載入任何數據。")
        else:
            st.error(f"所選測站 '{selected_station}' 的數據文件缺少參數 '{selected_param_display_original}' (原始列名: '{selected_param_col}')。")
            st.info(f"數據中可用的列: {df.columns.tolist()}")
        st.stop()

    st.info(f"正在對測站 **{selected_station}** 的參數 **{selected_param_display_original}** 執行 Transformer 預測...")

    # --- 數據預處理 ---
    df_processed = df[['ds', selected_param_col]].copy()
    df_processed.columns = ['ds', 'y']

    df_processed['ds'] = pd.to_datetime(df_processed['ds'])
    df_processed.sort_values('ds', inplace=True)

    # 根據選定的訓練日期範圍篩選數據
    train_start_datetime = pd.to_datetime(train_start_date)
    # 結束日期包含當天所有時間
    train_end_datetime = pd.to_datetime(train_end_date) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1) 

    df_processed = df_processed[
        (df_processed['ds'] >= train_start_datetime) &
        (df_processed['ds'] <= train_end_datetime)
    ].copy()

    if df_processed.empty:
        st.error(f"在選定的訓練時間範圍 ({train_start_date} 至 {train_end_date}) 內沒有找到數據。請調整時間範圍。")
        st.stop()
    
    # 移除重複的時間戳 (在重採樣前處理)
    if df_processed['ds'].duplicated().any():
        st.warning("警告：訓練數據中存在重複的時間戳，將移除重複項。")
        df_processed.drop_duplicates(subset=['ds'], keep='first', inplace=True)

    # 重採樣並計算平均值
    df_processed = df_processed.set_index('ds').resample(selected_freq_pandas).mean().reset_index()

    # 缺失值處理
    if missing_value_strategy == '前向填充 (ffill)':
        df_processed['y'] = df_processed['y'].ffill()
    elif missing_value_strategy == '後向填充 (bfill)':
        df_processed['y'] = df_processed['y'].bfill()
    elif missing_value_strategy == '線性插值 (interpolate)':
        df_processed['y'] = df_processed['y'].interpolate(method='linear')
    elif missing_value_strategy == '移除缺失值 (dropna)':
        df_processed = df_processed.dropna(subset=['y'])

    # 檢查處理後是否仍有有效數據
    if df_processed['y'].isnull().all():
        st.error(f"在經過預處理後，參數 '{selected_param_display}' 的數據全部為缺失值。無法進行預測。")
        st.stop()

    # 應用平滑
    if apply_smoothing and smoothing_window > 1:
        df_processed['y'] = df_processed['y'].rolling(window=smoothing_window, min_periods=1, center=True).mean()

    # 再次移除任何剩餘的 NaN (可能來自平滑或插值邊緣)
    df_processed.dropna(subset=['ds', 'y'], inplace=True)

    if df_processed.empty:
        st.error("經過數據預處理和時間範圍篩選後，沒有足夠的有效數據用於預測。請檢查原始數據、時間範圍和預處理選項。")
        st.stop()
    
    # 檢查數據長度是否滿足 look_back 要求
    if len(df_processed) <= look_back:
        st.error(f"訓練數據長度 ({len(df_processed)}) 必須大於回溯時間步長 ({look_back})。請增加數據範圍或減少回溯時間步長。")
        st.stop()

    # 數據歸一化 (0-1 範圍)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df_processed['y'].values.reshape(-1, 1))

    # 創建 Transformer 序列 (X, y)
    X, y = create_sequences(scaled_data, look_back)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1)) # 為 Transformer 增加特徵維度 (batch, sequence_length, features)

    # 訓練集和測試集劃分 (手動劃分)
    train_size = int(len(X) * (1 - validation_split))
    X_train, X_test = X[0:train_size,:], X[train_size:len(X),:]
    y_train, y_test = y[0:train_size], y[train_size:len(y)]

    # --- 建立並訓練 Transformer 模型 ---
    with st.spinner("正在建立並訓練 Transformer 模型..."):
        try:
            model = build_transformer_model(
                input_shape=(look_back, 1), # (sequence_length, features)
                head_size=head_size,
                num_heads=num_heads,
                ff_dim=ff_dim,
                num_transformer_blocks=num_transformer_blocks,
                mlp_units=mlp_units,
                dropout=dropout_rate
            )
            model.compile(optimizer='adam', loss='mean_squared_error')

            # 定義早停回調，防止過擬合
            early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

            # 訓練模型
            history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                                validation_data=(X_test, y_test), callbacks=[early_stopping], verbose=0)
        except Exception as e:
            st.error(f"Transformer 模型訓練失敗：{e}。請檢查數據或調整模型參數。")
            st.stop()

    st.success("Transformer 模型訓練完成！")

    ### 📚 訓練數據概覽
    st.subheader("📊 訓練數據概覽")
    if not df_processed.empty:
        total_duration = df_processed['ds'].max() - df_processed['ds'].min()
        total_records = len(df_processed)
        inferred_freq = None
        try:
            inferred_freq = pd.infer_freq(df_processed['ds'])
        except ValueError:
            inferred_freq = '無法精確推斷 (數據可能間隔不一致)'
        st.write(f"**使用數據區間**: 從 **{df_processed['ds'].min().strftime('%Y-%m-%d %H:%M')}** 到 **{df_processed['ds'].max().strftime('%Y-%m-%d %H:%M')}**")
        st.write(f"**總時長**: **{total_duration}**")
        st.write(f"**總筆數**: **{total_records}** 筆")
        st.write(f"**數據頻次 (預處理後)**: **{selected_freq_pandas}** (原始推斷: **{inferred_freq}**)")
    else:
        st.warning("沒有可用的訓練數據概覽。")

    ### 📊 數據品質概覽
    st.subheader("🔍 訓練數據品質報告")
    df_for_quality_check = df_processed.set_index('ds').rename(columns={'y': selected_param_col}).copy()
    quality_report = analyze_data_quality(df_for_quality_check, relevant_params=[selected_param_col])

    if selected_param_col in quality_report:
        metrics = quality_report[selected_param_col]
        st.write(f"**參數: {selected_param_display_original}**")
        st.write(f"- 總記錄數: {metrics['total_records']}")
        st.write(f"- 有效記錄數: {metrics['valid_count']}")
        st.write(f"- 缺失值數量: {metrics['missing_count']} (**{metrics['missing_percentage']:.2f}%**)")
        if metrics.get('is_numeric', True):
            st.write(f"- 零值數量: {metrics['zero_count']}")
            st.write(f"- 負值數量: {metrics['negative_count']}")
            st.write(f"- 潛在 IQR 異常值數量: {metrics['outlier_iqr_count']}")

            quality_data = {
                '類型': ['有效值', '缺失值', '零值', '負值', '潛在異常值'],
                '數量': [
                    metrics['valid_count'],
                    metrics['missing_count'],
                    metrics['zero_count'],
                    metrics['negative_count'],
                    metrics['outlier_iqr_count']
                ]
            }
            quality_df = pd.DataFrame(quality_data)
            quality_df = quality_df[quality_df['數量'] > 0] # 只顯示數量大於 0 的類別

            if not quality_df.empty:
                fig_quality = px.pie(
                    quality_df,
                    values='數量',
                    names='類型',
                    title=f"'{selected_param_display_original}' 數據品質分佈",
                    hole=0.3,
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig_quality.update_traces(textposition='inside', textinfo='percent+label', marker=dict(line=dict(color='#000000', width=1)))
                
                # 應用中文字體 (如果配置了)
                if CHINESE_FONT_PATH and os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', CHINESE_FONT_PATH)):
                    font_name = os.path.splitext(os.path.basename(CHINESE_FONT_PATH))[0]
                    fig_quality.update_layout(showlegend=True, font=dict(family=font_name))
                else:
                    fig_quality.update_layout(showlegend=True)
                
                st.plotly_chart(fig_quality, use_container_width=True)
            else:
                st.info("數據品質非常高，沒有缺失值、零值、負值或異常值。")
    else:
        st.warning(f"無法為參數 '{selected_param_display_original}' 生成數據品質報告。")


    ### 🎯 模型評估 (歷史數據)
    st.subheader("📉 模型性能評估")

    # 訓練集預測
    train_predict = model.predict(X_train)
    train_predict = scaler.inverse_transform(train_predict) # 反歸一化
    y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))

    # 測試集預測
    test_predict = model.predict(X_test)
    test_predict = scaler.inverse_transform(test_predict) # 反歸一化
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    # 計算 RMSE (均方根誤差)
    train_rmse = np.sqrt(mean_squared_error(y_train_actual, train_predict))
    test_rmse = np.sqrt(mean_squared_error(y_test_actual, test_predict))

    st.metric(label=f"訓練集 RMSE for {selected_param_display_original}", value=f"{train_rmse:.4f}")
    st.metric(label=f"測試集 RMSE for {selected_param_display_original}", value=f"{test_rmse:.4f}")
    st.info("**RMSE (均方根誤差)** 衡量模型在歷史數據上的預測誤差，值越小表示模型越精確。**測試集 RMSE** 反映模型對未見數據的泛化能力。")

    # 訓練過程中的損失曲線
    st.subheader("📈 模型訓練損失曲線")
    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(y=history.history['loss'], mode='lines', name='訓練損失'))
    if 'val_loss' in history.history: # 如果有驗證集，則顯示驗證損失
        fig_loss.add_trace(go.Scatter(y=history.history['val_loss'], mode='lines', name='驗證損失'))
    
    # 應用中文字體 (如果配置了)
    if CHINESE_FONT_PATH and os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', CHINESE_FONT_PATH)):
        font_name = os.path.splitext(os.path.basename(CHINESE_FONT_PATH))[0]
        fig_loss.update_layout(
            title="訓練與驗證損失",
            xaxis_title="Epoch",
            yaxis_title="損失 (MSE)",
            height=400,
            font=dict(family=font_name)
        )
    else:
        fig_loss.update_layout(
            title="訓練與驗證損失",
            xaxis_title="Epoch",
            yaxis_title="損失 (MSE)",
            height=400
        )
    st.plotly_chart(fig_loss, use_container_width=True)


    ### 📊 預測結果視覺化
    st.subheader("未來趨勢預測")

    # 未來預測：從最後 `look_back` 個點開始，迭代預測未來值
    # 確保 last_sequence 是 NumPy array，形狀為 (look_back, 1)
    last_sequence = scaled_data[-look_back:] 
    future_predictions = []

    with st.spinner(f"正在預測未來 {forecast_period_value} 個時間點..."):
        for _ in range(forecast_period_value):
            # 預測下一個時間點，模型輸入需要是 (1, look_back, 1)
            next_pred = model.predict(last_sequence.reshape(1, look_back, 1), verbose=0)[0, 0] # verbose=0 抑制預測時的進度條
            future_predictions.append(next_pred)
            # 更新序列：移除最舊的點，加入新的預測點
            last_sequence = np.append(last_sequence[1:], [[next_pred]], axis=0)

    # 反歸一化未來預測值
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

    # 生成未來時間戳
    last_known_date = df_processed['ds'].max()
    future_dates = pd.date_range(start=last_known_date + pd.to_timedelta(1, unit=selected_freq_pandas),
                                 periods=forecast_period_value,
                                 freq=selected_freq_pandas)

    forecast_df = pd.DataFrame({
        'ds': future_dates,
        'yhat': future_predictions.flatten()
    })

    # 合併所有結果用於繪圖
    full_plot_df = df_processed.copy()
    full_plot_df['yhat_train'] = np.nan
    full_plot_df['yhat_test'] = np.nan

    # 填充歷史預測值
    # 訓練預測的索引從 look_back 開始
    full_plot_df.loc[df_processed.index[look_back : look_back + len(train_predict)], 'yhat_train'] = train_predict.flatten()
    # 測試預測的索引從訓練預測之後開始
    full_plot_df.loc[df_processed.index[look_back + len(train_predict) : look_back + len(train_predict) + len(test_predict)], 'yhat_test'] = test_predict.flatten()

    fig = go.Figure()

    # 實際數據 (藍色實線)
    fig.add_trace(go.Scatter(
        x=full_plot_df['ds'],
        y=full_plot_df['y'],
        mode='lines',
        name='實際數據',
        line=dict(color='blue')
    ))

    # 訓練集預測 (綠色虛線)
    fig.add_trace(go.Scatter(
        x=full_plot_df['ds'],
        y=full_plot_df['yhat_train'],
        mode='lines',
        name='訓練集預測',
        line=dict(color='green', dash='dot')
    ))

    # 測試集預測 (橙色虛線)
    fig.add_trace(go.Scatter(
        x=full_plot_df['ds'],
        y=full_plot_df['yhat_test'],
        mode='lines',
        name='測試集預測',
        line=dict(color='orange', dash='dot')
    ))

    # 未來預測 (紅色虛線)
    fig.add_trace(go.Scatter(
        x=forecast_df['ds'],
        y=forecast_df['yhat'],
        mode='lines',
        name='未來預測',
        line=dict(color='red', dash='dash', width=2)
    ))

    forecast_unit_display = selected_prediction_freq_display.split(' ')[0]
    
    # 應用中文字體 (如果配置了)
    if CHINESE_FONT_PATH and os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', CHINESE_FONT_PATH)):
        font_name = os.path.splitext(os.path.basename(CHINESE_FONT_PATH))[0]
        fig.update_layout(
            title=f"{selected_station} - {selected_param_display_original} Transformer 未來 {forecast_period_value} {forecast_unit_display} 預測",
            xaxis_title="時間",
            yaxis_title=f"{selected_param_display_original} {param_unit}",
            hovermode="x unified",
            height=600,
            font=dict(family=font_name)
        )
    else:
        fig.update_layout(
            title=f"{selected_station} - {selected_param_display_original} Transformer 未來 {forecast_period_value} {forecast_unit_display} 預測",
            xaxis_title="時間",
            yaxis_title=f"{selected_param_display_original} {param_unit}",
            hovermode="x unified",
            height=600
        )
    st.plotly_chart(fig, use_container_width=True)

    ### 💾 下載預測結果
    st.markdown("您可以下載包含未來預測值的 CSV 文件。")

    download_df = forecast_df.copy()
    download_df.rename(columns={'ds': '時間', 'yhat': f'預測值_{selected_param_display_original}'}, inplace=True)
    download_df['時間'] = download_df['時間'].dt.strftime('%Y-%m-%d %H:%M:%S')

    csv_data = download_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="下載 Transformer 預測 CSV 文件",
        data=csv_data,
        file_name=f"{selected_station}_{selected_param_col}_Transformer_forecast.csv",
        mime="text/csv",
    )
