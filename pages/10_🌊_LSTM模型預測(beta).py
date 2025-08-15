import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots 
import os
import io
import json
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr 
import zipfile

# --- 新增 ---
import joblib
import hashlib

from utils.helpers import get_station_name_from_id, initialize_session_state, load_data

pio.templates.default = "plotly_white"

# --- 嘗試導入 TensorFlow / Keras ---
tensorflow_available = False
try:
    import tensorflow as tf
    from keras.models import Sequential, Model
    from keras.layers import LSTM, Dense, Dropout, Input 
    from keras.callbacks import EarlyStopping, Callback 
    from keras.models import load_model # --- 新增 ---
    tensorflow_available = True
except ImportError:
    st.error("錯誤：TensorFlow/Keras 庫未安裝或無法載入。LSTM 模型預測功能將無法使用。")
    st.info("若需使用此功能，請在您的 Python 環境中運行以下命令：")
    st.code("pip install tensorflow scikit-learn numpy plotly scipy joblib")
    st.warning("確保您的 TensorFlow 安裝與您的系統和 CUDA 版本兼容 (如果使用 GPU)。")

# --- 新增：模型快取輔助函式 ---
def analyze_data_quality(df, relevant_params):
    report = {}
    for param_col in relevant_params:
        if param_col not in df.columns:
            continue
        
        s = df[param_col]
        total_records = len(s)
        missing_count = s.isnull().sum()
        valid_count = total_records - missing_count
        
        is_numeric = pd.api.types.is_numeric_dtype(s)
        param_metrics = {
            'total_records': total_records, 'valid_count': valid_count, 'missing_count': missing_count,
            'missing_percentage': (missing_count / total_records * 100) if total_records > 0 else 0,
            'is_numeric': is_numeric
        }

        if is_numeric and valid_count > 0:
            param_metrics['zero_count'] = (s == 0).sum()
            param_metrics['negative_count'] = (s < 0).sum()
            Q1 = s.quantile(0.25)
            Q3 = s.quantile(0.75)
            IQR = Q3 - Q1
            if IQR > 0:
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                param_metrics['outlier_iqr_count'] = ((s < lower_bound) | (s > upper_bound)).sum()
            else:
                param_metrics['outlier_iqr_count'] = 0
        else:
             param_metrics['zero_count'] = 0
             param_metrics['negative_count'] = 0
             param_metrics['outlier_iqr_count'] = 0

        report[param_col] = param_metrics
    return report

def assess_risk(value, param_key):
    """根據預測值、參數名稱和設定檔，回傳風險等級"""
    # 從 config 中讀取 risk_thresholds 區塊，如果找不到則回傳空字典
    thresholds = st.session_state['risk_thresholds'].get(param_key, {})
    
    # 如果 config 中沒有設定此參數的閾值，則回傳"未知"
    if not thresholds:
        return "未知"
    
    # 讀取危險和警告等級，如果沒有設定，則設為無限大
    danger_level = thresholds.get("danger", float('inf'))
    warning_level = thresholds.get("warning", float('inf'))

    if value >= danger_level:
        return "危險"
    elif value >= warning_level:
        return "警告"
    else:
        return "安全"
def get_local_model_paths(parameters: dict):
    """根據參數字典生成唯一的模型、scaler和history路徑"""
    model_dir = "trained_models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    config_str = "".join([f"{k}:{v}" for k, v in sorted(parameters.items())])
    model_hash = hashlib.md5(config_str.encode()).hexdigest()
    
    model_path = os.path.join(model_dir, f"lstm_model_{model_hash}.keras")
    scaler_path = os.path.join(model_dir, f"lstm_scaler_{model_hash}.joblib")
    # 新增 history 路徑
    history_path = os.path.join(model_dir, f"lstm_history_{model_hash}.json")
    
    return model_path, scaler_path, history_path

def save_local_model(model, scaler, history_data: dict, parameters: dict):
    """保存模型、scaler 和訓練歷史"""
    try:
        model_path, scaler_path, history_path = get_local_model_paths(parameters)
        model.save(model_path)
        joblib.dump(scaler, scaler_path)
        # 將 history 字典存成 json
        with open(history_path, 'w') as f:
            json.dump(history_data, f)
        st.toast(f"模型與訓練歷史已成功快取！")
    except Exception as e:
        st.warning(f"儲存模型快取時發生錯誤: {e}")

def load_local_model(parameters: dict):
    """嘗試載入已保存的模型、scaler 和訓練歷史"""
    model_path, scaler_path, history_path = get_local_model_paths(parameters)
    
    # 確認三個檔案都存在
    if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(history_path):
        try:
            tf.keras.backend.clear_session()
            model = load_model(model_path)
            scaler = joblib.load(scaler_path)
            # 讀取 history json
            with open(history_path, 'r') as f:
                history_data = json.load(f)
            return model, scaler, history_data
        except Exception as e:
            st.warning(f"載入快取模型時發生錯誤: {e}。將重新進行訓練。")
            return None, None, None
            
    return None, None, None

# --- 新增：將 AccuracyHistory 類別的定義移到頂層 ---

class StreamlitProgressBar(Callback):
    """一個用於在 Streamlit 中顯示 Keras 訓練進度的 Callback。"""
    def __init__(self, epochs):
        super().__init__()
        self.epochs = epochs
        # 在 Streamlit 介面上建立一個進度條元件和一個空的文字位置
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # 計算目前進度
        progress = (epoch + 1) / self.epochs
        
        # 更新進度條的長度
        self.progress_bar.progress(progress)
        
        # 從 logs 字典中獲取最新的 loss 值並顯示
        loss = logs.get('loss', 0)
        val_loss = logs.get('val_loss', 0)
        status = f"Epoch {epoch + 1}/{self.epochs} - Loss: {loss:.4f}, Val Loss: {val_loss:.4f}"
        
        # 更新狀態文字
        self.status_text.text(status)

    def on_train_end(self, logs=None):
        # 訓練結束後，清除進度條和文字，保持介面乾淨
        self.progress_bar.empty()
        self.status_text.empty()
class AccuracyHistory(Callback):
    def __init__(self, X_train, y_train, X_test, y_test, scaler, epsilon, look_back):
        super().__init__()
        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test
        self.scaler, self.epsilon = scaler, epsilon
        # 這些列表現在主要用於繪製即時的訓練曲線
        self.train_accuracies, self.val_accuracies = [], []
        self.train_correlations, self.val_correlations = [], []

    # 1. 將所有計算邏輯移到一個新函式中，並接收 model 作為參數
    def calculate_metrics(self, model):
        """手動計算並記錄一次準確率和相關係數"""
        train_pred_scaled = model.predict(self.X_train, verbose=0)
        train_actual_scaled = self.y_train.reshape(-1, 1)
        train_pred_original = self.scaler.inverse_transform(train_pred_scaled)
        train_actual_original = self.scaler.inverse_transform(train_actual_scaled)
        train_correct_count = np.sum(np.abs(train_pred_original - train_actual_original) <= self.epsilon)
        self.train_accuracies.append(train_correct_count / len(train_actual_original) if len(train_actual_original) > 0 else 0)
        if len(train_actual_original) > 1 and len(train_pred_original.flatten()) > 1:
            train_corr, _ = pearsonr(train_actual_original.flatten(), train_pred_original.flatten())
            self.train_correlations.append(train_corr)
        else: self.train_correlations.append(np.nan)

        val_pred_scaled = model.predict(self.X_test, verbose=0)
        val_actual_scaled = self.y_test.reshape(-1, 1)
        val_pred_original = self.scaler.inverse_transform(val_pred_scaled)
        val_actual_original = self.scaler.inverse_transform(val_actual_scaled)
        val_correct_count = np.sum(np.abs(val_pred_original - val_actual_original) <= self.epsilon)
        self.val_accuracies.append(val_correct_count / len(val_actual_original) if len(val_actual_original) > 0 else 0)
        if len(val_actual_original) > 1 and len(val_pred_original.flatten()) > 1:
            val_corr, _ = pearsonr(val_actual_original.flatten(), val_pred_original.flatten())
            self.val_correlations.append(val_corr)
        else: self.val_correlations.append(np.nan)

    # 2. 讓 on_epoch_end 在訓練時呼叫這個新函式
    def on_epoch_end(self, epoch, logs=None):
        """此函式由 Keras 在每個 epoch 結束時自動呼叫"""
        if logs is None:
            logs = {}
        
        # 執行計算
        self.calculate_metrics(self.model)
        
        # <<< 核心修改：將最新的計算結果回寫到 logs 中 >>>
        # 這樣 Keras 就會自動把這些指標記錄到 history.history 物件裡
        logs['train_accuracy'] = self.train_accuracies[-1]
        logs['val_accuracy'] = self.val_accuracies[-1]
        logs['train_correlation'] = self.train_correlations[-1]
        logs['val_correlation'] = self.val_correlations[-1]
# --- 設定頁面 ---
st.set_page_config(
    page_title="LSTM 模型預測",
    page_icon="🌊",
    layout="wide"
)
initialize_session_state()

st.title("🌊 海洋數據 LSTM 模型預測")
st.markdown("使用長短期記憶 (LSTM) 類神經網絡預測海洋數據的未來趨勢。")

predictable_params_config_map = {
    col_name: info["display_zh"] for col_name, info in st.session_state.get('parameter_info', {}).items()
    if info.get("type") == "linear"
}

def create_sequences(data, look_back):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), 0])
        y.append(data[i + look_back, 0])
    return np.array(X), np.array(y)

# --- 側邊欄：LSTM 預測設定控制項 ---
st.sidebar.header("LSTM 預測設定")

locations = st.session_state.get('locations', [])

if not locations:
    st.sidebar.warning("請在 `config.json` 的 `STATION_COORDS` 中配置測站資訊。")
    st.stop()

selected_station = st.sidebar.selectbox("選擇測站:", locations, key='pages_10_lstm_station', format_func=get_station_name_from_id)
selected_station_name = get_station_name_from_id(selected_station)
df_initial_check = load_data(selected_station, st.session_state.get('parameter_info', {}))

available_predictable_params_display_to_col = {}
if not df_initial_check.empty:
    for col_name_config, display_name in predictable_params_config_map.items():
        param_info_for_check = st.session_state['parameter_info'].get(col_name_config, {})
        expected_names = [col_name_config.lower()]
        if "column_name_in_data" in param_info_for_check: expected_names.append(param_info_for_check["column_name_in_data"].lower())
        if "display_zh" in param_info_for_check: expected_names.append(param_info_for_check["display_zh"].lower())
        
        target_col_to_check = next((name for name in expected_names if name in df_initial_check.columns), None)
        
        if target_col_to_check and pd.api.types.is_numeric_dtype(df_initial_check[target_col_to_check]) and df_initial_check[target_col_to_check].count() > 0:
            available_predictable_params_display_to_col[display_name] = target_col_to_check

if not available_predictable_params_display_to_col:
    st.sidebar.error("載入數據後，沒有可供預測的有效數值型參數。")
    st.stop()

selected_param_display = st.sidebar.selectbox("選擇預測參數:", list(available_predictable_params_display_to_col.keys()), key='pages_10_lstm_param_display', format_func=lambda x: x if x in available_predictable_params_display_to_col else "未知參數")
selected_param_col = available_predictable_params_display_to_col[selected_param_display]

param_unit = next((info.get("unit", "") for key, info in st.session_state.get('parameter_info', {}).items() if key == selected_param_col), "")

st.sidebar.markdown("---")
st.sidebar.subheader("預測時間設定")
prediction_frequencies = {"小時 (H)": "h", "天 (D)": "D", "週 (W)": "W", "月 (M)": "ME", "年 (Y)": "YE"}
selected_prediction_freq_display = st.sidebar.selectbox("選擇預測頻次:", list(prediction_frequencies.keys()), key='pages_10_prediction_frequency')
selected_freq_pandas = prediction_frequencies[selected_prediction_freq_display]
forecast_period_value = st.sidebar.number_input(f"預測未來多久 ({selected_prediction_freq_display.split(' ')[0]}):",min_value=1, value=24, step=1, key='pages_10_forecast_period_value')
st.sidebar.markdown("---")
st.sidebar.subheader("訓練數據時間範圍")

if not df_initial_check.empty and 'ds' in df_initial_check.columns and not df_initial_check['ds'].isnull().all():
    min_date_available = df_initial_check['ds'].min().date()
    max_date_available = df_initial_check['ds'].max().date()
else:
    min_date_available = pd.to_datetime('1990-01-01').date()
    max_date_available = pd.Timestamp.now().date()

train_start_date = st.sidebar.date_input("訓練數據開始日期:", value=min_date_available, min_value=min_date_available, max_value=max_date_available, key='pages_10_train_start_date')
train_end_date = st.sidebar.date_input("訓練數據結束日期:", value=max_date_available, min_value=min_date_available, max_value=max_date_available, key='pages_10_train_end_date')

if train_start_date >= train_end_date:
    st.sidebar.error("訓練數據開始日期必須早於結束日期。")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.subheader("數據預處理")
missing_value_strategy = st.sidebar.selectbox("缺失值處理:", options=['前向填充 (ffill)', '後向填充 (bfill)', '線性插值 (interpolate)', '移除缺失值 (dropna)'], key='pages_10_missing_strategy')
apply_smoothing = st.sidebar.checkbox("應用數據平滑", value=False, key='pages_10_apply_smoothing')
smoothing_window = 1
if apply_smoothing:
    smoothing_window = st.sidebar.slider("平滑處理 (移動平均視窗):", min_value=1, max_value=24, value=3, step=1)
st.sidebar.markdown("---")
st.sidebar.subheader("模型準確率設定")
epsilon_value = st.sidebar.number_input("準確率 ε 誤差區間:", min_value=0.001, max_value=10.0, value=0.1, step=0.01, format="%.3f", help="設定一個誤差範圍 ε。當 |預測值 - 實際值| <= ε 時，此預測被視為「正確」。", key='pages_10_epsilon_value')
st.sidebar.markdown("---")
st.sidebar.subheader("LSTM 模型參數")
look_back = st.sidebar.slider("回溯時間步 (look_back):", 1, 48, 6, 1)
lstm_units = st.sidebar.slider("LSTM 層單元數:", 10, 200, 50, 10)
epochs = st.sidebar.number_input("訓練迭代次數 (Epochs):", 10, 500, 50, 10)
batch_size = st.sidebar.number_input("批次大小 (Batch Size):", 1, 128, 32, 8)
dropout_rate = st.sidebar.slider("Dropout 比率:", 0.0, 0.5, 0.2, 0.05)
validation_split = st.sidebar.slider("驗證集比例:", 0.0, 0.5, 0.1, 0.05)
patience = st.sidebar.number_input("早停耐心值 (Patience):", min_value=5, max_value=200, value=50, step=5)

if st.sidebar.button("🌊 執行 LSTM 預測"):
    if not tensorflow_available:
        st.error("TensorFlow/Keras 庫不可用，無法執行 LSTM 預測。")
        st.stop()

    if df_initial_check.empty or selected_param_col not in df_initial_check.columns:
        st.error(f"所選測站 '{selected_station_name}' 的數據文件缺少參數 '{selected_param_display}'。")
        st.stop()
    
    with st.spinner("STEP 1/3: 正在預處理數據..."):
        df_processed = df_initial_check[['ds', selected_param_col]].copy()
        df_processed.columns = ['ds', 'y']
        
        train_start_datetime = pd.to_datetime(train_start_date)
        train_end_datetime = pd.to_datetime(train_end_date) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        df_processed = df_processed[(df_processed['ds'] >= train_start_datetime) & (df_processed['ds'] <= train_end_datetime)].copy()
        
        if df_processed.empty:
            st.error(f"在選定的訓練時間範圍 ({train_start_date} 至 {train_end_date}) 內沒有找到數據。")
            st.stop()
            
        df_processed = df_processed.set_index('ds').resample(selected_freq_pandas).mean().reset_index()
        
        if missing_value_strategy == '前向填充 (ffill)': df_processed['y'] = df_processed['y'].ffill()
        elif missing_value_strategy == '後向填充 (bfill)': df_processed['y'] = df_processed['y'].bfill()
        elif missing_value_strategy == '線性插值 (interpolate)': df_processed['y'] = df_processed['y'].interpolate(method='linear')
        
        if apply_smoothing and smoothing_window > 1:
            df_processed['y'] = df_processed['y'].rolling(window=smoothing_window, min_periods=1, center=True).mean()
            
        df_processed.dropna(subset=['ds', 'y'], inplace=True)
        
        if df_processed.empty or len(df_processed) <= look_back:
            st.error(f"經過數據預處理後，沒有足夠的有效數據用於預測 (長度需大於 {look_back})。")
            st.stop()

    model_params = {
        "page": "lstm_prediction", "station": selected_station_name, "param": selected_param_col,
        "freq": selected_freq_pandas, "look_back": look_back, "lstm_units": lstm_units, "epochs": epochs,
        "batch_size": batch_size, "dropout": dropout_rate, "smoothing": smoothing_window if apply_smoothing else 0,
        "start_date": train_start_date.strftime('%Y-%m-%d'), "end_date": train_end_date.strftime('%Y-%m-%d'),
        "missing_strategy": missing_value_strategy
    }

    model, scaler, history_data = load_local_model(model_params)
    history = None 

    if model is None:
        st.info("🛠️ 未找到快取模型，開始新的訓練...")
        st.write("---") 
        st.write("**STEP 2/3: 正在訓練 LSTM 模型...**") # 用一般文字標題取代 spinner
        
        # 數據縮放與塑形 (這部分程式碼不變)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df_processed['y'].values.reshape(-1, 1))
        X, y = create_sequences(scaled_data, look_back)
        train_size = int(len(X) * (1 - validation_split))
        X_train, X_test = X[0:train_size,:], X[train_size:len(X),:]
        y_train, y_test = y[0:train_size], y[train_size:len(y)]
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        
        # 建立模型 (這部分程式碼不變)
        accuracy_history_callback = AccuracyHistory(X_train, y_train, X_test, y_test, scaler, epsilon_value, look_back)
        model = Sequential([
            Input(shape=(look_back, 1)),
            LSTM(lstm_units, return_sequences=True), Dropout(dropout_rate),
            LSTM(lstm_units, return_sequences=False), Dropout(dropout_rate),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        
        # <<< 核心修改點 >>>
        # 1. 在這裡建立我們的進度條物件
        progress_callback = StreamlitProgressBar(epochs)
        
        try:
            # 2. 將 progress_callback 加入 callbacks 列表
            history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                                validation_data=(X_test, y_test), 
                                callbacks=[early_stopping, accuracy_history_callback, progress_callback], 
                                verbose=0)
            history_data = history.history
        except Exception as e:
            st.error(f"LSTM 模型訓練失敗：{e}")
            st.stop()
            
        st.success("模型訓練完成！")
        save_local_model(model, scaler, history_data, model_params)
    else:
        st.success("✅ 成功載入快取模型！")
        with st.spinner("STEP 2/3: 正在準備數據..."):
            scaled_data = scaler.transform(df_processed['y'].values.reshape(-1, 1))
            X, y = create_sequences(scaled_data, look_back)
            train_size = int(len(X) * (1 - validation_split))
            X_train, X_test = X[0:train_size,:], X[train_size:len(X),:]
            y_train, y_test = y[0:train_size], y[train_size:len(y)]
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
            accuracy_history_callback = AccuracyHistory(X_train, y_train, X_test, y_test, scaler, epsilon_value, look_back)

    with st.spinner("STEP 3/3: 正在評估與視覺化..."):
        st.subheader("📚 訓練數據概覽")
        if not df_processed.empty:
            total_duration = df_processed['ds'].max() - df_processed['ds'].min()
            total_records = len(df_processed)
            inferred_freq = None
            try:
                # 使用 .copy() 避免潛在的 SettingWithCopyWarning
                inferred_freq = pd.infer_freq(df_processed['ds'].copy())
            except ValueError:
                inferred_freq = '無法精確推斷 (數據可能間隔不一致)'
            
            st.write(f"**使用數據區間**: 從 **{df_processed['ds'].min().strftime('%Y-%m-%d %H:%M')}** 到 **{df_processed['ds'].max().strftime('%Y-%m-%d %H:%M')}**")
            st.write(f"**總時長**: **{total_duration}**")
            st.write(f"**總筆數**: **{total_records}** 筆")
            st.write(f"**數據頻次 (預處理後)**: **{selected_freq_pandas}** (原始推斷: **{inferred_freq or 'N/A'}**)")
        else:
            st.warning("沒有可用的訓練數據概覽。")

        st.subheader("📊 數據品質概覽")
        # 我們要分析的是經過時間範圍篩選和預處理後的 df_processed
        df_for_quality_check = df_processed.rename(columns={'y': selected_param_col})
        quality_report = analyze_data_quality(df_for_quality_check, [selected_param_col])

        if selected_param_col in quality_report:
            metrics = quality_report[selected_param_col]
            
            # 使用 st.columns 讓排版更好看
            c1, c2 = st.columns([1, 2])
            with c1:
                st.write(f"**分析參數: {selected_param_display}**")
                st.metric("總筆數", metrics['total_records'])
                st.metric("有效筆數", metrics['valid_count'])
                st.metric("缺失比例", f"{metrics['missing_percentage']:.2f}%")
                st.metric("潛在異常值", metrics['outlier_iqr_count'])
            
            with c2:
                quality_data = {
                    '類型': ['有效值', '缺失值', '零值', '負值', '潛在異常值'], 
                    '數量': [
                        metrics['valid_count'], metrics['missing_count'], 
                        metrics['zero_count'], metrics['negative_count'], 
                        metrics['outlier_iqr_count']
                    ]
                }
                quality_df = pd.DataFrame(quality_data)
                quality_df = quality_df[quality_df['數量'] > 0]

                if not quality_df.empty:
                    # 修正：使用 selected_param_display
                    fig_quality = px.pie(quality_df, values='數量', names='類型', 
                                         title=f"'{selected_param_display}' 數據品質分佈", 
                                         hole=0.3, color_discrete_sequence=px.colors.qualitative.Pastel)
                    fig_quality.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig_quality, use_container_width=True, key="quality_chart")
                else:
                    st.info("數據品質非常高，沒有偵測到缺失、零、負值或異常值。")
        st.write("---")
        st.subheader("📉 模型性能評估")
        train_predict = model.predict(X_train)
        train_predict = scaler.inverse_transform(train_predict)
        y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
        test_predict = model.predict(X_test)
        test_predict = scaler.inverse_transform(test_predict)
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
        train_rmse = np.sqrt(mean_squared_error(y_train_actual, train_predict))
        test_rmse = np.sqrt(mean_squared_error(y_test_actual, test_predict))
        
        accuracy_history_callback.calculate_metrics(model)
        final_train_accuracy = accuracy_history_callback.train_accuracies[-1]
        final_val_accuracy = accuracy_history_callback.val_accuracies[-1]
        train_corr = accuracy_history_callback.train_correlations[-1]
        test_corr = accuracy_history_callback.val_correlations[-1]
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("訓練集表現")
            st.metric(label=f"RMSE", value=f"{train_rmse:.4f}")
            st.metric(label=f"相關係數 (R)", value=f"{train_corr:.4f}")
            st.metric(label=f"準確率 (ε={epsilon_value:.2f})", value=f"{final_train_accuracy:.2%}")
        with col2:
            st.subheader("測試/驗證集表現")
            st.metric(label=f"RMSE", value=f"{test_rmse:.4f}")
            st.metric(label=f"相關係數 (R)", value=f"{test_corr:.4f}")
            st.metric(label=f"準確率 (ε={epsilon_value:.2f})", value=f"{final_val_accuracy:.2%}")

        st.subheader("模型訓練過程評估曲線")
        if history_data:
            fig_loss_acc = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08, 
                                         subplot_titles=("訓練與驗證損失 (MSE)", 
                                                         f"訓練與驗證準確率 (ε={epsilon_value})", 
                                                         "訓練與驗證相關係數 (R)"))
            if 'loss' in history_data:
                fig_loss_acc.add_trace(go.Scatter(y=history_data['loss'], mode='lines', name='訓練損失'), row=1, col=1)
            if 'val_loss' in history_data:
                fig_loss_acc.add_trace(go.Scatter(y=history_data['val_loss'], mode='lines', name='驗證損失'), row=1, col=1)
            if 'train_accuracy' in history_data:
                fig_loss_acc.add_trace(go.Scatter(y=history_data['train_accuracy'], mode='lines', name='訓練準確率'), row=2, col=1)
            if 'val_accuracy' in history_data:
                fig_loss_acc.add_trace(go.Scatter(y=history_data['val_accuracy'], mode='lines', name='驗證準確率'), row=2, col=1)
            if 'train_correlation' in history_data:
                fig_loss_acc.add_trace(go.Scatter(y=history_data['train_correlation'], mode='lines', name='訓練相關係數'), row=3, col=1)
            if 'val_correlation' in history_data:
                fig_loss_acc.add_trace(go.Scatter(y=history_data['val_correlation'], mode='lines', name='驗證相關係數'), row=3, col=1)
            
            fig_loss_acc.update_layout(height=800, hovermode="x unified")
            st.plotly_chart(fig_loss_acc, use_container_width=True, key="loss_chart")
        else:
            st.info("找不到與此模型關聯的訓練歷史數據。")

        # --- 預測結果視覺化與風險評估的正確順序 ---

        st.subheader("📈 預測結果視覺化")
        
        # 1. 先準備好所有繪圖需要的 DataFrame
        last_sequence = scaled_data[-look_back:]
        future_predictions = []
        for _ in range(forecast_period_value):
            next_pred = model.predict(last_sequence.reshape(1, look_back, 1), verbose=0)[0, 0]
            future_predictions.append(next_pred)
            last_sequence = np.append(last_sequence[1:], [[next_pred]], axis=0)
        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
        last_known_date = df_processed['ds'].max()
        
        if selected_freq_pandas in ['ME', 'YE']:
            future_start_date = last_known_date
        else:
            future_start_date = last_known_date + pd.to_timedelta(1, unit=selected_freq_pandas[0])

        future_dates = pd.date_range(start=future_start_date, periods=forecast_period_value, freq=selected_freq_pandas)
        forecast_df = pd.DataFrame({'ds': future_dates, 'yhat': future_predictions.flatten()})

        train_dates = df_processed['ds'][look_back : look_back + len(train_predict)].reset_index(drop=True)
        train_predict_df = pd.DataFrame({
            'ds': train_dates,
            'yhat_train': train_predict.flatten()
        })
        
        test_start_index = look_back + len(train_predict)
        test_end_index = test_start_index + len(test_predict)
        test_dates = df_processed['ds'][test_start_index : test_end_index].reset_index(drop=True)
        test_predict_df = pd.DataFrame({
            'ds': test_dates,
            'yhat_test': test_predict.flatten()
        })
        
        # 2. 建立圖表物件並畫圖
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_processed['ds'], y=df_processed['y'], mode='lines', name='實際數據'))
        fig.add_trace(go.Scatter(x=train_predict_df['ds'], y=train_predict_df['yhat_train'], mode='lines', name='訓練集預測', line=dict(dash='dot')))
        fig.add_trace(go.Scatter(x=test_predict_df['ds'], y=test_predict_df['yhat_test'], mode='lines', name='測試集預測', line=dict(dash='dot')))
        fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat'], mode='lines', name='未來預測', line=dict(dash='dash')))
        fig.update_layout(title=f"{selected_station_name} - {selected_param_display} LSTM 未來 {forecast_period_value} {selected_prediction_freq_display.split(' ')[0]} 預測", xaxis_title="時間", yaxis_title=f"{selected_param_display} {param_unit}", hovermode="x unified", height=600)
        st.plotly_chart(fig, use_container_width=True, key="forecast_chart")

        # 3. 在圖表下方，顯示風險評估結果
        st.write("---")
        st.subheader("航行風險評估 (基於預測)")

        # <<< 修正順序：第一步，先找出對應的 key >>>
        param_key_in_config = next((key for key, info in st.session_state.get('parameter_info', {}).items() 
                                    if info.get("display_zh") == selected_param_display), None)
        
        # <<< 修正順序：第二步，用這個 key 產生說明文字 >>>
        explanation_text = "此評估基於 `config.json` 中設定的風險閾值。"
        if param_key_in_config:
            thresholds = st.session_state.get('parameter_info', {}).get(param_key_in_config, {}).get("risk_thresholds", {})
            if thresholds:
                warning_level = thresholds.get("warning")
                danger_level = thresholds.get("danger")
                explanation_parts = []
                if warning_level is not None:
                    explanation_parts.append(f"**警告**等級 > {warning_level} {param_unit}")
                if danger_level is not None:
                    explanation_parts.append(f"**危險**等級 > {danger_level} {param_unit}")
                if explanation_parts:
                    explanation_text = f"此評估根據為 **{selected_param_display}** 設定的風險閾值：{'，'.join(explanation_parts)}。當所有預測值均低於「警告」標準時，評估結果為安全。"
        
        st.info(explanation_text, icon="ℹ️")

        # <<< 修正順序：第三步，用這個 key 執行風險評估 >>>
        if param_key_in_config:
            forecast_df['risk_level'] = forecast_df['yhat'].apply(
                lambda value: assess_risk(value, param_key_in_config)
            )
        else:
            forecast_df['risk_level'] = "未知"

        # 顯示總體風險結論
        if "危險" in forecast_df['risk_level'].unique():
            st.error(f"**綜合評估：未來 {forecast_period_value} {selected_prediction_freq_display.split(' ')[0]} 內存在「危險」等級風險！**")
        elif "警告" in forecast_df['risk_level'].unique():
            st.warning(f"**綜合評估：未來 {forecast_period_value} {selected_prediction_freq_display.split(' ')[0]} 內存在「警告」等級風險。**")
        else:
            st.success(f"**綜合評估：未來 {forecast_period_value} {selected_prediction_freq_display.split(' ')[0]} 內預測均在安全範圍內。**")

        # 顯示高風險時段詳情
        risky_forecasts = forecast_df[forecast_df['risk_level'].isin(['警告', '危險'])].copy()
        if not risky_forecasts.empty:
            with st.expander("查看詳細高風險時段"):
                risky_forecasts['ds'] = risky_forecasts['ds'].dt.strftime('%Y-%m-%d %H:%M')
                risky_forecasts['yhat'] = risky_forecasts['yhat'].map('{:.2f}'.format)
                st.dataframe(
                    risky_forecasts[['ds', 'yhat', 'risk_level']].rename(columns={
                        'ds': '時間', 'yhat': f'預測值 ({param_unit})', 'risk_level': '風險等級'
                    }),
                    use_container_width=True, hide_index=True
                )
        st.subheader("💾 下載預測結果與報告")
        
        st.markdown("您可以下載包含預測值的 CSV 文件、互動式圖表，或一份包含所有執行參數與結果的完整報告。")

        col1, col2, col3 = st.columns(3)

        with col1:
            download_df = forecast_df.copy()
            download_df.rename(columns={'ds': '時間', 'yhat': f'預測值_{selected_param_display}'}, inplace=True)
            download_df['時間'] = download_df['時間'].dt.strftime('%Y-%m-%d %H:%M:%S')
            csv_data = download_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="下載預測數據 (CSV)",
                data=csv_data,
                file_name=f"{selected_station_name}_{selected_param_col}_LSTM_forecast_data.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col2:
            fig_data = fig.to_html(full_html=True, include_plotlyjs='cdn')
            fig_bytes = fig_data.encode('utf-8')

            st.download_button(
                label="下載預測圖表 (HTML)",
                data=fig_bytes,
                file_name=f"{selected_station_name}_{selected_param_col}_LSTM_forecast_chart.html",
                mime="text/html",
                use_container_width=True
            )
            
        with col3:
        # <<< 簡化並修正判斷邏輯 >>>
            actual_epochs_text = f"(實際執行: {len(history_data['loss'])})" if history_data else "(從快取載入)"

            report_content = f"""
    # LSTM 時間序列預測報告
    ## 測站: {selected_station_name}
    ## 預測參數: {selected_param_display} ({param_unit})

    ---
    ## 1. 數據概覽與設定
    - **使用數據區間**: {df_processed['ds'].min().strftime('%Y-%m-%d %H:%M')} 到 {df_processed['ds'].max().strftime('%Y-%m-%d %H:%M')}
    - **總時長**: {df_processed['ds'].max() - df_processed['ds'].min()}
    - **總筆數 (預處理後)**: {len(df_processed)} 筆
    - **預測頻次**: {selected_prediction_freq_display}

    ---
    ## 2. 數據預處理設定
    - **缺失值處理策略**: {missing_value_strategy}
    - **數據平滑處理**: {'是' if apply_smoothing and smoothing_window > 1 else '否'}
    """
            if apply_smoothing and smoothing_window > 1:
                report_content += f"   - **移動平均視窗**: {smoothing_window}\n"

            report_content += f"""
    ---
    ## 3. LSTM 模型參數
    - **回溯時間步 (look_back)**: {look_back}
    - **LSTM 層單元數**: {lstm_units}
    - **訓練迭代次數 (Epochs)**: {epochs} {actual_epochs_text}
    - **批次大小 (Batch Size)**: {batch_size}
    - **Dropout 比率**: {dropout_rate}
    - **驗證集比例**: {validation_split:.2f}
    - **早停耐心值 (Patience)**: {patience}

    ---
    ## 4. 模型性能評估 (最終)
    ### 訓練集表現
    - **RMSE**: {train_rmse:.4f}
    - **相關係數 (R)**: {train_corr:.4f}
    - **準確率 (ε={epsilon_value:.2f})**: {final_train_accuracy:.2%}

    ### 測試/驗證集表現
    - **RMSE**: {test_rmse:.4f}
    - **相關係數 (R)**: {test_corr:.4f}
    - **準確率 (ε={epsilon_value:.2f})**: {final_val_accuracy:.2%}

    ---
    報告生成時間: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
    """
            st.download_button(
                label="下載完整報告 (TXT)",
                data=report_content.encode('utf-8'),
                file_name=f"{selected_station_name}_{selected_param_col}_LSTM_report.txt",
                mime="text/plain",
                use_container_width=True,
                help="下載包含所有執行參數與結果的文本報告"
            )

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f"{selected_station_name}_{selected_param_col}_LSTM_forecast_data.csv", csv_data)
            zf.writestr(f"{selected_station_name}_{selected_param_col}_LSTM_report.txt", report_content.encode('utf-8'))
            zf.writestr(f"{selected_station_name}_{selected_param_col}_LSTM_forecast_chart.html", fig_bytes)
        st.download_button("🚀 一鍵打包下載 (ZIP)", zip_buffer.getvalue(), f"{selected_station_name}_{selected_param_col}_LSTM_forecast_package.zip",
                           "application/zip", use_container_width=True,
                           help="下載包含預測數據、圖表和報告的壓縮包")
