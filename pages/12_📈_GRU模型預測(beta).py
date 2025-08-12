import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Dropout, GRU 
from keras.callbacks import EarlyStopping, Callback
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 從 helpers 模組導入所有必要的通用函數和全局變數
# 假設 helpers.py 中有這些函數
from utils.helpers import (
    get_station_name_from_id,
    initialize_session_state,
    load_app_config_and_font, 
    load_data_for_prediction_page, 
    create_sequences, 
    PARAMETER_INFO, 
    BASE_DATA_PATH_FROM_CONFIG,
    CHINESE_FONT_NAME,
    load_year_data,
    get_available_years 
)

# 設置 TensorFlow 日誌級別，抑制 INFO 訊息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# --- Streamlit 頁面設定 ---
st.set_page_config(
    page_title="GRU 模型預測",
    page_icon="📈",
    layout="wide"
)
initialize_session_state()

st.title("📈 海洋數據 GRU 模型預測")
st.markdown("使用門控循環單元 (GRU) 類神經網絡預測海洋數據的未來趨勢。")

# --- 載入配置 ---
try:
    app_config = load_app_config_and_font()
    STATION_COORDS = app_config.get("STATION_COORDS", {})
except Exception as e:
    st.error(f"無法載入應用程式配置：{e}")
    st.stop()

# --- GRU 模型輔助函數 ---
def build_gru_model(input_shape, gru_units, dense_units, num_gru_layers, dropout=0.2):
    inputs = Input(shape=input_shape)
    x = inputs
    for i in range(num_gru_layers):
        return_sequences = True if i < num_gru_layers - 1 else False
        x = GRU(gru_units, return_sequences=return_sequences, dropout=dropout)(x)
    for dim in dense_units:
        x = Dense(dim, activation="relu")(x)
        x = Dropout(dropout)(x)
    outputs = Dense(1)(x)
    return Model(inputs=inputs, outputs=outputs)


# --- 側邊欄：GRU 預測設定控制項 ---
st.sidebar.header("GRU 預測設定")

locations = st.session_state.get('locations', [])
if not locations:
    st.sidebar.warning("請在 `config.json` 的 `STATION_COORDS` 中配置測站資訊。")
    st.stop()

selected_station = st.sidebar.selectbox("選擇測站:", locations, key='pages_12_gru_station', format_func=get_station_name_from_id)
selected_station_name = get_station_name_from_id(selected_station)

predictable_params_config_map = {
    col_name: info["display_zh"] for col_name, info in PARAMETER_INFO.items()
    if info.get("type") == "linear"
}

# 動態獲取可用參數
available_predictable_params_display_to_col = {}
if selected_station:
    current_year = pd.Timestamp.now().year
    temp_base_path = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', BASE_DATA_PATH_FROM_CONFIG))
    temp_df_for_col_check = None
    for y_check in range(current_year, current_year - 5, -1):
        temp_df_for_col_check = load_year_data(temp_base_path, selected_station, y_check)
        if temp_df_for_col_check is not None and not temp_df_for_col_check.empty: break
    
    if temp_df_for_col_check is not None and not temp_df_for_col_check.empty:
        for col_name, display_name in predictable_params_config_map.items():
            if col_name in temp_df_for_col_check.columns and pd.api.types.is_numeric_dtype(temp_df_for_col_check[col_name]):
                available_predictable_params_display_to_col[display_name] = col_name
    else:
        st.sidebar.warning(f"無法為測站 '{selected_station_name}' 載入任何歷史數據以確認可用參數。請檢查數據檔案。")

if not available_predictable_params_display_to_col:
    st.sidebar.error("沒有可供預測的有效數值型參數。")
    st.stop()

selected_param_display = st.sidebar.selectbox("選擇預測參數:", list(available_predictable_params_display_to_col.keys()), key='pages_12_gru_param_display')
selected_param_col = available_predictable_params_display_to_col[selected_param_display]
param_info_original = PARAMETER_INFO.get(selected_param_col, {})
selected_param_display_original = param_info_original.get("display_zh", selected_param_col)
param_unit = param_info_original.get("unit", "")

st.sidebar.markdown("---")
st.sidebar.subheader("預測時間設定")
# 修改 prediction_frequencies 字典，使用 'ME' 和 'YE'
prediction_frequencies = {"小時 (H)": "h", "天 (D)": "D", "週 (W)": "W", "月 (M)": "ME", "年 (Y)": "YE"}
selected_prediction_freq_display = st.sidebar.selectbox("選擇預測頻次:", list(prediction_frequencies.keys()), key='pages_12_prediction_frequency')
selected_freq_pandas = prediction_frequencies[selected_prediction_freq_display]

# 根據新的頻率別名調整 max_forecast_value_map
max_forecast_value_map = {'h': 24 * 30, 'D': 365, 'W': 104, 'ME': 24, 'YE': 5} # 使用 'ME' 和 'YE'
max_forecast_value = max_forecast_value_map.get(selected_freq_pandas, 30)
default_forecast_value = 24 if selected_freq_pandas == 'h' else 7 if selected_freq_pandas == 'D' else 4 if selected_freq_pandas == 'ME' else 1 # 調整預設值

forecast_period_value = st.sidebar.number_input(f"預測未來多久 ({selected_prediction_freq_display.split(' ')[0]}):", 1, max_forecast_value, min(default_forecast_value, max_forecast_value), 1, key='pages_12_forecast_period_value')

st.sidebar.markdown("---")
st.sidebar.subheader("訓練數據時間範圍")
available_years = get_available_years(BASE_DATA_PATH_FROM_CONFIG, locations)
if not available_years:
    st.sidebar.error("沒有可用的數據年份。")
    st.stop()
min_year_available, max_year_available = min(available_years), max(available_years)
min_date_available, max_date_available = pd.to_datetime(f'{min_year_available}-01-01').date(), pd.to_datetime(f'{max_year_available}-12-31').date()
default_start_date = max(min_date_available, max_date_available - pd.Timedelta(days=365))
train_start_date = st.sidebar.date_input("訓練數據開始日期:", default_start_date, min_date_available, max_date_available, key='pages_12_train_start_date')
train_end_date = st.sidebar.date_input("訓練數據結束日期:", max_date_available, min_date_available, max_date_available, key='pages_12_train_end_date')
if train_start_date >= train_end_date:
    st.sidebar.error("訓練數據開始日期必須早於結束日期。")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.subheader("數據預處理")
missing_value_strategy = st.sidebar.selectbox("缺失值處理:", ['前向填充 (ffill)', '後向填充 (bfill)', '線性插值 (interpolate)', '移除缺失值 (dropna)'], key='pages_12_missing_strategy')
apply_smoothing = st.sidebar.checkbox("應用數據平滑", False, key='pages_12_apply_smoothing')
if apply_smoothing:
    smoothing_window = st.sidebar.slider("平滑處理 (移動平均視窗):", 1, 24, 3, 1, help="移動平均視窗大小。", key='pages_12_smoothing_window')
else:
    smoothing_window = 1

st.sidebar.markdown("---")
st.sidebar.subheader("數據正規化")
normalization_method = st.sidebar.selectbox("選擇正規化方法:", ['Min-Max 歸一化 (0-1)', '標準化 (Z-score)', 'RobustScaler (中位數-四分位距)'], key='pages_12_normalization_method')

st.sidebar.markdown("---")
st.sidebar.subheader("模型準確率設定")
epsilon_value = st.sidebar.number_input("準確率 ε 誤差區間:", 0.001, 10.0, 0.1, 0.01, "%.3f", help="設定一個誤差範圍 ε。當 |預測值 - 實際值| <= ε 時，此預測被視為「正確」。", key='pages_12_epsilon_value')

st.sidebar.markdown("---")
st.sidebar.subheader("GRU 模型參數")
look_back = st.sidebar.slider("回溯時間步 (look_back):", 1, 48, 12, 1, help="GRU 模型考慮多少個過去的時間點。", key='pages_12_gru_look_back')
gru_units = st.sidebar.slider("GRU 層單元數:", 32, 256, 64, 32, help="GRU 層的神經元數量。", key='pages_12_gru_units')
num_gru_layers = st.sidebar.slider("GRU 層數量:", 1, 5, 2, 1, help="堆疊的 GRU 層數量。", key='pages_12_num_gru_layers')
mlp_units = st.sidebar.multiselect("MLP 層單元數:", [32, 64, 128, 256], [128], help="預測頭部的多層感知器層。", key='pages_12_mlp_units')
epochs = st.sidebar.number_input("訓練迭代次數 (Epochs):", 10, 500, 100, 10, key='pages_12_epochs')
batch_size = st.sidebar.number_input("批次大小 (Batch Size):", 1, 128, 32, 8, key='pages_12_batch_size')
dropout_rate = st.sidebar.slider("Dropout 比率:", 0.0, 0.5, 0.2, 0.05, help="防止過擬合。", key='pages_12_dropout_rate')
validation_split = st.sidebar.slider("驗證集比例:", 0.0, 0.5, 0.1, 0.05, help="用於模型驗證的數據比例。", key='pages_12_validation_split')
# 修改早停耐心值的 max_value 和 value
patience = st.sidebar.number_input("早停耐心值 (Patience):", min_value=5, max_value=200, value=50, step=5, help="驗證損失在多少個 epochs 內沒有改善則停止訓練。", key='pages_12_patience')

# --- 修改：增強 AccuracyHistory 以包含相關係數 ---
class AccuracyHistory(Callback):
    def __init__(self, X_train, y_train, X_test, y_test, scaler, epsilon):
        super().__init__()
        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test
        self.scaler, self.epsilon = scaler, epsilon
        self.train_accuracies, self.val_accuracies = [], []
        self.train_correlations, self.val_correlations = [], [] # 新增
    
    def on_epoch_end(self, epoch, logs=None):
        # 訓練集
        train_pred_scaled = self.model.predict(self.X_train, verbose=0)
        train_actual_scaled = self.y_train.reshape(-1, 1)
        train_pred_original = self.scaler.inverse_transform(train_pred_scaled)
        train_actual_original = self.scaler.inverse_transform(train_actual_scaled)
        self.train_accuracies.append(np.sum(np.abs(train_pred_original - train_actual_original) <= self.epsilon) / len(train_actual_original))
        if len(train_actual_original) > 1:
            train_corr, _ = pearsonr(train_actual_original.flatten(), train_pred_original.flatten())
            self.train_correlations.append(train_corr)
        else:
            self.train_correlations.append(np.nan)
        
        # 驗證集
        val_pred_scaled = self.model.predict(self.X_test, verbose=0)
        val_actual_scaled = self.y_test.reshape(-1, 1)
        val_pred_original = self.scaler.inverse_transform(val_pred_scaled)
        val_actual_original = self.scaler.inverse_transform(val_actual_scaled)
        self.val_accuracies.append(np.sum(np.abs(val_pred_original - val_actual_original) <= self.epsilon) / len(val_actual_original))
        if len(val_actual_original) > 1:
            val_corr, _ = pearsonr(val_actual_original.flatten(), val_pred_original.flatten())
            self.val_correlations.append(val_corr)
        else:
            self.val_correlations.append(np.nan)

# --- 執行預測按鈕 ---
if st.sidebar.button("📈 執行 GRU 預測"):
    if not tf.config.list_physical_devices('GPU'):
        st.warning("警告: TensorFlow 未啟用 GPU 加速。模型訓練可能較慢。")

    with st.spinner("正在載入數據..."):
        df_raw = load_data_for_prediction_page(selected_station, selected_param_col, train_start_date, train_end_date)
    
    if df_raw.empty:
        st.error(f"無法為測站 '{selected_station_name}' 在指定時間範圍內載入參數 '{selected_param_display_original}' 的數據。")
        st.stop()
    
    st.info(f"正在對測站 **{selected_station_name}** 的參數 **{selected_param_display_original}** 執行 GRU 預測...")

    # --- 數據預處理 ---
    df_processed = df_raw.copy().sort_values('ds').drop_duplicates(subset=['ds'], keep='first')
    df_processed = df_processed.set_index('ds').resample(selected_freq_pandas).mean()
    if missing_value_strategy == '前向填充 (ffill)': df_processed['y'] = df_processed['y'].ffill()
    elif missing_value_strategy == '後向填充 (bfill)': df_processed['y'] = df_processed['y'].bfill()
    elif missing_value_strategy == '線性插值 (interpolate)': df_processed['y'] = df_processed['y'].interpolate(method='linear')
    elif missing_value_strategy == '移除缺失值 (dropna)': df_processed.dropna(subset=['y'], inplace=True)
    if df_processed['y'].isnull().all(): st.error("數據預處理後全部為空值。"); st.stop()
    if apply_smoothing and smoothing_window > 1: df_processed['y'] = df_processed['y'].rolling(smoothing_window, min_periods=1, center=True).mean()
    df_processed.dropna(subset=['y'], inplace=True)
    df_processed.reset_index(inplace=True) 
    if len(df_processed) <= look_back: st.error(f"有效數據點 ({len(df_processed)}) 不足，無法進行訓練 (需要 > {look_back})。"); st.stop()
    
    # --- 正規化與序列創建 ---
    if normalization_method.startswith('Min-Max'): scaler = MinMaxScaler(feature_range=(0, 1))
    elif normalization_method.startswith('標準化'): scaler = StandardScaler()
    else: scaler = RobustScaler()
    scaled_data = scaler.fit_transform(df_processed['y'].values.reshape(-1, 1))
    X, y = create_sequences(scaled_data, look_back)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1)) 
    train_size = int(len(X) * (1 - validation_split))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # --- 模型訓練 ---
    with st.spinner("正在建立並訓練 GRU 模型..."):
        model = build_gru_model((look_back, 1), gru_units, mlp_units, num_gru_layers, dropout_rate)
        model.compile(optimizer='adam', loss='mean_squared_error')
        early_stopping = EarlyStopping('val_loss', patience=patience, restore_best_weights=True)
        accuracy_history_callback = AccuracyHistory(X_train, y_train, X_test, y_test, scaler, epsilon_value)
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), callbacks=[early_stopping, accuracy_history_callback], verbose=0)
    st.success("GRU 模型訓練完成！")

    # --- 數據概覽與品質報告 ---
    st.subheader("📊 數據概覽與品質分析")
    # ... (這部分可以從您的 LSTM 程式碼中複製過來，這裡暫時省略以保持簡潔)

    # --- 修改：重構模型性能評估區塊 ---
    st.subheader("📉 模型性能評估")
    train_predict = scaler.inverse_transform(model.predict(X_train, verbose=0))
    y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
    test_predict = scaler.inverse_transform(model.predict(X_test, verbose=0))
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    train_rmse = np.sqrt(mean_squared_error(y_train_actual, train_predict))
    test_rmse = np.sqrt(mean_squared_error(y_test_actual, test_predict))
    train_corr, _ = pearsonr(y_train_actual.flatten(), train_predict.flatten()) if len(y_train_actual) > 1 else (np.nan, np.nan)
    test_corr, _ = pearsonr(y_test_actual.flatten(), test_predict.flatten()) if len(y_test_actual) > 1 else (np.nan, np.nan)
    final_train_accuracy = accuracy_history_callback.train_accuracies[-1] if accuracy_history_callback.train_accuracies else np.nan
    final_val_accuracy = accuracy_history_callback.val_accuracies[-1] if accuracy_history_callback.val_accuracies else np.nan
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("訓練集表現")
        st.metric("RMSE", f"{train_rmse:.4f}", help="均方根誤差，值越小越好。")
        st.metric("相關係數 (R)", f"{train_corr:.4f}", help="衡量趨勢吻合度，值越接近 1 越好。")
        st.metric(f"準確率 (ε={epsilon_value:.2f})", f"{final_train_accuracy:.2%}", help=f"在 ±{epsilon_value:.2f} 誤差內的正確率。")
    with col2:
        st.subheader("測試/驗證集表現")
        st.metric("RMSE", f"{test_rmse:.4f}", help="對未見數據的泛化誤差，值越小越好。")
        st.metric("相關係數 (R)", f"{test_corr:.4f}", help="對未見數據的趨勢吻合度，值越接近 1 越好。")
        st.metric(f"準確率 (ε={epsilon_value:.2f})", f"{final_val_accuracy:.2%}", help=f"在 ±{epsilon_value:.2f} 誤差內的正確率。")

    # --- 修改：增強訓練過程圖表 ---
    st.subheader("📈 模型訓練過程評估曲線")
    fig_loss_acc = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=("訓練與驗證損失 (MSE)", f"訓練與驗證準確率 (ε={epsilon_value})", "訓練與驗證相關係數 (R)"))
    fig_loss_acc.add_trace(go.Scatter(y=history.history['loss'], name='訓練損失'), row=1, col=1)
    if 'val_loss' in history.history: fig_loss_acc.add_trace(go.Scatter(y=history.history['val_loss'], name='驗證損失'), row=1, col=1)
    fig_loss_acc.add_trace(go.Scatter(y=accuracy_history_callback.train_accuracies, name='訓練準確率', line=dict(color='green')), row=2, col=1)
    fig_loss_acc.add_trace(go.Scatter(y=accuracy_history_callback.val_accuracies, name='驗證準確率', line=dict(color='red')), row=2, col=1)
    fig_loss_acc.add_trace(go.Scatter(y=accuracy_history_callback.train_correlations, name='訓練相關係數', line=dict(color='purple')), row=3, col=1)
    fig_loss_acc.add_trace(go.Scatter(y=accuracy_history_callback.val_correlations, name='驗證相關係數', line=dict(color='cyan')), row=3, col=1)
    fig_loss_acc.update_layout(height=800, xaxis_title="Epoch", yaxis1=dict(title="損失 (MSE)"), yaxis2=dict(title="準確率"), yaxis3=dict(title="相關係數 (R)"), hovermode="x unified", font=dict(family=CHINESE_FONT_NAME))
    st.plotly_chart(fig_loss_acc, use_container_width=True)

    # --- 預測結果視覺化 ---
    st.subheader("📊 未來趨勢預測")
    last_sequence = scaled_data[-look_back:]
    future_predictions = []
    for _ in range(forecast_period_value):
        next_pred = model.predict(last_sequence.reshape(1, look_back, 1), verbose=0)[0, 0]
        future_predictions.append(next_pred)
        last_sequence = np.append(last_sequence[1:], [[next_pred]], axis=0)
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    last_known_date = df_processed['ds'].max()

    # Determine the start date for the future prediction range based on frequency type
    # Check for calendar-based frequencies (Month End, Year End)
    if selected_freq_pandas in ['ME', 'YE']: # <-- 修改判斷條件，檢查 'ME' 和 'YE'
        # For monthly or yearly frequencies, pd.date_range starts *after* the start date
        # So, we just use the last known date as the start reference.
        # pd.date_range with freq='ME' or 'YE' will correctly generate dates at the end of each period.
        future_start_date = last_known_date
    else:
        # For fixed frequencies (hourly, daily, weekly), add one unit to get the next point
        # This is where pd.to_timedelta is appropriate
        future_start_date = last_known_date + pd.to_timedelta(1, unit=selected_freq_pandas)

    future_dates = pd.date_range(start=future_start_date, periods=forecast_period_value, freq=selected_freq_pandas)
    forecast_df = pd.DataFrame({'ds': future_dates, 'yhat': future_predictions.flatten()})

    # 準備繪圖數據
    full_plot_df = df_processed.copy()
    full_plot_df['yhat_train'] = np.nan
    full_plot_df.loc[df_processed.index[look_back:len(train_predict) + look_back], 'yhat_train'] = train_predict.flatten()
    full_plot_df['yhat_test'] = np.nan
    full_plot_df.loc[df_processed.index[len(train_predict) + look_back:], 'yhat_test'] = test_predict.flatten()
    
    # 繪圖
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=full_plot_df['ds'], y=full_plot_df['y'], name='實際數據', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=full_plot_df['ds'], y=full_plot_df['yhat_train'], name='訓練集預測', line=dict(color='green', dash='dot')))
    fig.add_trace(go.Scatter(x=full_plot_df['ds'], y=full_plot_df['yhat_test'], name='測試集預測', line=dict(color='orange', dash='dot')))
    fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat'], name='未來預測', line=dict(color='red', dash='dash')))
    fig.update_layout(title=f"{selected_station_name} - {selected_param_display_original} GRU 預測", xaxis_title="時間", yaxis_title=f"{selected_param_display_original} {param_unit}", height=600, font=dict(family=CHINESE_FONT_NAME))
    st.plotly_chart(fig, use_container_width=True)

    # --- 修改：擴充下載功能 ---
    st.subheader("💾 下載預測結果與報告")
    st.markdown("您可以下載預測數據、互動式圖表或一份包含所有執行參數與結果的完整報告。")
    col1, col2, col3 = st.columns(3)
    with col1:
        csv_data = forecast_df.rename(columns={'ds': '時間', 'yhat': f'預測值_{selected_param_display_original}'}).to_csv(index=False).encode('utf-8')
        st.download_button("下載預測數據 (CSV)", csv_data, f"{selected_station_name}_{selected_param_col}_GRU_data.csv", "text/csv", use_container_width=True)
    with col2:
        html_bytes = fig.to_html(full_html=True, include_plotlyjs='cdn').encode('utf-8')
        st.download_button("下載預測圖表 (HTML)", html_bytes, f"{selected_station_name}_{selected_param_col}_GRU_chart.html", "text/html", use_container_width=True)
    with col3:
        report_content = f"""
# GRU 時間序列預測報告
## 測站: {selected_station_name} | 預測參數: {selected_param_display_original} ({param_unit})
---
## 1. 數據與預測設定
- 數據區間: {train_start_date.strftime('%Y-%m-%d')} 到 {train_end_date.strftime('%Y-%m-%d')}
- 預測頻次: {selected_prediction_freq_display}
- 預測未來時長: {forecast_period_value} {selected_prediction_freq_display.split(' ')[0]}
- 缺失值處理: {missing_value_strategy}
- 數據平滑: {'是 (窗口: ' + str(smoothing_window) + ')' if apply_smoothing and smoothing_window > 1 else '否'}
- 正規化方法: {normalization_method}
---
## 2. GRU 模型參數
- 回溯時間步 (look_back): {look_back}
- GRU 層數量: {num_gru_layers}
- GRU 層單元數: {gru_units}
- MLP 層單元數: {mlp_units}
- 訓練迭代次數 (Epochs): {epochs} (實際執行: {len(history.history['loss'])})
- 批次大小: {batch_size}
- Dropout 比率: {dropout_rate}
- 驗證集比例: {validation_split:.2f}
- 早停耐心值: {patience}
---
## 3. 模型性能評估
### 訓練集表現
- RMSE: {train_rmse:.4f}
- 相關係數 (R): {train_corr:.4f}
- 準確率 (ε={epsilon_value:.2f}): {final_train_accuracy:.2%}
### 測試/驗證集表現
- RMSE: {test_rmse:.4f}
- 相關係數 (R): {test_corr:.4f}
- 準確率 (ε={epsilon_value:.2f}): {final_val_accuracy:.2%}
---
報告生成時間: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        st.download_button("下載完整報告 (TXT)", report_content.encode('utf-8'), f"{selected_station_name}_{selected_param_col}_GRU_report.txt", "text/plain", use_container_width=True, help="下載包含所有設定與結果的文本報告")
