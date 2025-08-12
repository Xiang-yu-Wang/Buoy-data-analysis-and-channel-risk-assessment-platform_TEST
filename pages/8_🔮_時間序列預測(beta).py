import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots # 確保 make_subplots 被導入
import os
import itertools
from glob import glob
import json
import plotly.express as px
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from utils.helpers import get_station_name_from_id, initialize_session_state, load_data
from scipy.stats import pearsonr 
import plotly.io as pio 
import logging 

# --- 引入新的異常值檢測庫 ---
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from scipy import stats
from statsmodels.tsa.seasonal import STL 
# --- 引入新的異常值檢測庫結束 ---

# --- 嘗試導入 Prophet 及相關庫 ---
prophet_available = False
try:
    from prophet import Prophet
    from prophet.plot import plot_plotly
    from prophet.diagnostics import cross_validation, performance_metrics 
    from sklearn.model_selection import ParameterGrid 
    prophet_available = True
    # --- 修正點 3: 設置 cmdstanpy 的日誌級別，減少 INFO 輸出 ---
    prophet_logger = logging.getLogger('cmdstanpy')
    prophet_logger.setLevel(logging.WARNING)
except ImportError:
    st.error("錯誤：Prophet 庫或其依賴未安裝或無法載入。Prophet 模型預測功能將無法使用。")
    st.info("若需使用此功能，請在您的 Python 環境中運行以下命令：")
    st.code("pip install prophet scikit-learn numpy plotly scipy")
    st.warning("對於 Prophet，它可能還需要底層的 C++ 編譯器 (如 CmdStan)。請參考 Prophet 官方文檔進行安裝。")


# --- 嘗試導入 statsmodels 及 pmdarima ---
statsmodels_available = False
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.api import ExponentialSmoothing
    import pmdarima as pm # For auto_arima
    statsmodels_available = True
except ImportError:
    st.error("錯誤：statsmodels 或 pmdarima 庫未安裝或無法載入。SARIMA 和 ETS 模型預測功能將無法使用。")
    st.info("若需使用此功能，請在您的 Python 環境中運行以下命令：")
    st.code("pip install statsmodels pmdarima")


# --- 設定頁面 ---
st.set_page_config(
    page_title="時間序列預測 (Beta)",
    page_icon="🔮",
    layout="wide"
)
initialize_session_state()

st.title("🔮 海洋數據時間序列預測 (Beta)")
st.markdown("使用 Prophet、SARIMA 或 ETS 模型預測海洋數據的未來趨勢。")

locations = st.session_state.get('locations', [])

predictable_params_config_map = {
    col_name: info["display_zh"] for col_name, info in st.session_state.get('parameter_info', {}).items()
    if info.get("type") == "linear"
}


# --- 輔助函數：計算布林帶 ---
def calculate_bollinger_bands(df, window=20, num_std_dev=2):
    if len(df) < window:
        return None
    
    df_temp = df.copy() 
    df_temp['MA'] = df_temp['y'].rolling(window=window).mean()
    df_temp['StdDev'] = df_temp['y'].rolling(window=window).std()
    df_temp['Upper'] = df_temp['MA'] + (df_temp['StdDev'] * num_std_dev)
    df_temp['Lower'] = df_temp['MA'] - (df_temp['StdDev'] * num_std_dev)
    return df_temp

# --- 輔助函數：數據品質分析 ---
def analyze_data_quality(df_to_check, relevant_params):
    report = {}
    for param in relevant_params:
        if param not in df_to_check.columns:
            continue
        
        s = df_to_check[param]
        total_records = len(s)
        missing_count = s.isnull().sum()
        valid_count = total_records - missing_count
        
        is_numeric = pd.api.types.is_numeric_dtype(s)
        
        param_metrics = {
            'total_records': total_records,
            'valid_count': valid_count,
            'missing_count': missing_count,
            'missing_percentage': (missing_count / total_records * 100) if total_records > 0 else 0,
            'is_numeric': is_numeric
        }

        if is_numeric:
            s_numeric = s.dropna()
            param_metrics['zero_count'] = (s_numeric == 0).sum()
            param_metrics['negative_count'] = (s_numeric < 0).sum()
            
            # --- 修正點：確保 outlier_iqr_count 在任何情況下都被定義 ---
            param_metrics['outlier_iqr_count'] = 0 # 先給定預設值為 0
            if not s_numeric.empty:
                Q1 = s_numeric.quantile(0.25)
                Q3 = s_numeric.quantile(0.75)
                IQR = Q3 - Q1
                
                if IQR > 0: # 只有當 IQR 大於 0 時，才計算上下限和異常值
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    param_metrics['outlier_iqr_count'] = ((s_numeric < lower_bound) | (s_numeric > upper_bound)).sum()
                # 如果 IQR <= 0，outlier_iqr_count 就會保持預設的 0，這是合理的。
            
            param_metrics['min_val'] = s_numeric.min() if not s_numeric.empty else np.nan
            param_metrics['max_val'] = s_numeric.max() if not s_numeric.empty else np.nan
            param_metrics['mean_val'] = s_numeric.mean() if not s_numeric.empty else np.nan
            param_metrics['std_val'] = s_numeric.std() if not s_numeric.empty else np.nan
        
        report[param] = param_metrics
    return report

# --- 輔助函數：異常值檢測與處理 ---
def detect_outliers(df, param, method='iqr', iqr_multiplier=1.5, z_threshold=3, if_contamination='auto', n_neighbors=20, stl_seasonal_period_input=None, selected_freq_pandas_input='D'):
    """
    檢測時間序列中的異常值。
    """
    s = df[param].copy().dropna() 
    is_outlier = pd.Series(False, index=s.index)

    if s.empty:
        return pd.Series(False, index=df.index) 

    if method == 'iqr':
        Q1 = s.quantile(0.25)
        Q3 = s.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - iqr_multiplier * IQR
        upper_bound = Q3 + iqr_multiplier * IQR
        is_outlier = (s < lower_bound) | (s > upper_bound)
    elif method == 'zscore':
        if s.std() == 0: 
            is_outlier = pd.Series(False, index=s.index)
        else:
            z = np.abs(stats.zscore(s))
            is_outlier = z > z_threshold
    elif method == 'modified_zscore':
        median = s.median()
        median_abs_dev = np.median(np.abs(s - median))
        if median_abs_dev == 0: 
            is_outlier = pd.Series(False, index=s.index)
        else:
            modified_z = 0.6745 * (s - median) / median_abs_dev
            is_outlier = np.abs(modified_z) > z_threshold
    elif method == 'isolation_forest':
        if len(s) < 2:
            is_outlier = pd.Series(False, index=s.index)
        else:
            try:
                contamination_val = 'auto' if if_contamination == 'auto' else float(if_contamination)
                if_model = IsolationForest(contamination=contamination_val, random_state=42)
                if_model.fit(s.values.reshape(-1, 1))
                is_outlier = if_model.predict(s.values.reshape(-1, 1)) == -1
            except Exception as e:
                st.warning(f"Isolation Forest 檢測失敗: {e}. 將跳過此方法。")
                is_outlier = pd.Series(False, index=s.index)
    elif method == 'lof':
        if len(s) < n_neighbors + 1:
            is_outlier = pd.Series(False, index=s.index)
        else:
            try:
                lof_model = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=False) 
                is_outlier = lof_model.fit_predict(s.values.reshape(-1, 1)) == -1
            except Exception as e:
                st.warning(f"LOF 檢測失敗: {e}. 將跳過此方法。")
                is_outlier = pd.Series(False, index=s.index)
    elif method == 'stl_residual':
        stl_seasonal_period = stl_seasonal_period_input
        if stl_seasonal_period is None or stl_seasonal_period <= 1:
            if selected_freq_pandas_input == 'h': stl_seasonal_period = 24
            elif selected_freq_pandas_input == 'D': stl_seasonal_period = 7 
            elif selected_freq_pandas_input == 'W': stl_seasonal_period = 52 
            elif selected_freq_pandas_input == 'M': stl_seasonal_period = 12
            elif selected_freq_pandas_input == 'Q': stl_seasonal_period = 4
            elif selected_freq_pandas_input == 'Y': stl_seasonal_period = 1 
            else: stl_seasonal_period = 13 
            if stl_seasonal_period <= 1 and 'seasonal' in df.columns: 
                st.warning(f"STL 季節性週期自動設定為 {stl_seasonal_period}，若數據有明顯季節性請檢查預測頻次或手動調整。")


        if len(s) < 2 * stl_seasonal_period or stl_seasonal_period <= 1:
            if stl_seasonal_period > 1: 
                st.warning(f"數據點 ({len(s)}) 不足，無法進行 STL 分解 (需要至少 {2 * stl_seasonal_period} 點)。將跳過此方法。")
            is_outlier = pd.Series(False, index=s.index)
        else:
            try:
                stl = STL(s, seasonal=stl_seasonal_period, period=stl_seasonal_period, robust=True)
                res = stl.fit()
                residual = res.resid.dropna() 
                if residual.std() == 0:
                    is_outlier = pd.Series(False, index=s.index)
                else:
                    is_outlier = np.abs(residual - residual.mean()) > z_threshold * residual.std()
            except Exception as e:
                st.warning(f"STL 分解檢測失敗: {e}. 將跳過此方法。")
                is_outlier = pd.Series(False, index=s.index)

    full_outlier_series = pd.Series(False, index=df.index)
    full_outlier_series[is_outlier.index] = is_outlier
    return full_outlier_series.fillna(False)


def handle_outliers(df, param, is_outlier, strategy='replace_interpolate'):
    """
    處理時間序列中的異常值。
    """
    df_processed = df.copy()
    
    if strategy == 'remove':
        df_processed = df_processed[~is_outlier].reset_index(drop=True)
    elif strategy == 'interpolate':
        df_processed.loc[is_outlier, param] = np.nan
        df_processed[param] = df_processed[param].interpolate(method='linear')
        df_processed[param] = df_processed[param].ffill().bfill()
    elif strategy == 'cap':
        Q1 = df_processed[param].quantile(0.25)
        Q3 = df_processed[param].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR 
        upper_bound = Q3 + 3 * IQR
        df_processed[param] = df_processed[param].clip(lower_bound, upper_bound)
    elif strategy == 'mark':
        pass 
    
    return df_processed

# --- 側邊欄：預測設定控制項 ---
st.sidebar.header("時間序列預測設定")

if not locations:
    st.sidebar.warning("請在 `config.json` 的 `STATION_COORDS` 中配置測站資訊。")
    st.stop()

selected_station = st.sidebar.selectbox("選擇測站:", locations, key='pages_8_station', format_func=get_station_name_from_id)

# 預載入數據以動態獲取可用參數
df_initial_check = load_data(selected_station, st.session_state.get('parameter_info', {}))

available_predictable_params_display_to_col = {}
for col_name, display_name in predictable_params_config_map.items():
    param_col_in_data = st.session_state.get('parameter_info', {}).get(col_name, {}).get("column_name_in_data", col_name).lower()
    
    if param_col_in_data in df_initial_check.columns and pd.api.types.is_numeric_dtype(df_initial_check[param_col_in_data]):
        if df_initial_check[param_col_in_data].count() > 0: 
            available_predictable_params_display_to_col[display_name] = param_col_in_data

if not available_predictable_params_display_to_col:
    st.sidebar.error("載入數據後，沒有可供預測的有效數值型參數。請檢查數據文件和 `config.json` 中的參數配置。")
    st.stop()

selected_param_display = st.sidebar.selectbox("選擇預測參數:", list(available_predictable_params_display_to_col.keys()), key='pages_8_param_display')
selected_param_col = available_predictable_params_display_to_col[selected_param_display]

param_info_original = {}
for key, val in st.session_state.get('parameter_info', {}).items():
    if val.get("column_name_in_data", key).lower() == selected_param_col:
        param_info_original = val
        break

selected_param_display_original = param_info_original.get("display_zh", selected_param_col)
param_unit = param_info_original.get("unit", "")


st.sidebar.markdown("---")
st.sidebar.subheader("預測時間設定")

prediction_frequencies = {
    "小時 (h)": "h",
    "天 (D)": "D",
    "週 (W)": "W",
    "月 (M)": "M",
    "年 (Y)": "Y"
}
selected_prediction_freq_display = st.sidebar.selectbox(
    "選擇預測頻次:",
    list(prediction_frequencies.keys()),
    key='pages_8_prediction_frequency'
)
selected_freq_pandas = prediction_frequencies[selected_prediction_freq_display] 

forecast_period_value = st.sidebar.number_input(
    f"預測未來多久 ({selected_prediction_freq_display.split(' ')[0]}):",
    min_value=1,
    max_value=365 if selected_freq_pandas == 'D' else 8760 if selected_freq_pandas == 'h' else 12, 
    value=24 if selected_freq_pandas == 'h' else 7 if selected_freq_pandas == 'D' else 1,
    step=1,
    key='pages_8_forecast_period_value'
)

# --- 數據訓練時間範圍選擇 ---
st.sidebar.markdown("---")
st.sidebar.subheader("訓練數據時間範圍")

if not df_initial_check.empty and 'ds' in df_initial_check.columns and not df_initial_check['ds'].isnull().all():
    min_date_available = df_initial_check['ds'].min().date()
    max_date_available = df_initial_check['ds'].max().date()
else:
    min_date_available = pd.to_datetime('1990-01-01').date()
    max_date_available = pd.Timestamp.now().date()
    st.warning("無法從載入的數據中獲取時間範圍。使用預設日期範圍。")

default_start_date = min_date_available
default_end_date = max_date_available

train_start_date = st.sidebar.date_input(
    "訓練數據開始日期:",
    value=default_start_date,
    min_value=min_date_available,
    max_value=max_date_available,
    key='pages_8_train_start_date'
)
train_end_date = st.sidebar.date_input(
    "訓練數據結束日期:",
    value=default_end_date,
    min_value=min_date_available,
    max_value=max_date_available,
    key='pages_8_train_end_date'
)

if train_start_date >= train_end_date:
    st.sidebar.error("訓練數據開始日期必須早於結束日期。")
    st.stop()


# --- 數據預處理選項 ---
st.sidebar.markdown("---")
st.sidebar.subheader("數據預處理")
missing_value_strategy = st.sidebar.selectbox(
    "缺失值處理:",
    options=['前向填充 (ffill)', '後向填充 (bfill)', '線性插值 (interpolate)', '移除缺失值 (dropna)', '不處理 (保持原樣)'],
    key='pages_8_missing_strategy'
)

# --- 進階異常值處理選項 ---
st.sidebar.markdown("---")
st.sidebar.subheader("進階異常值處理")
outlier_method = st.sidebar.selectbox(
    "異常值檢測方法:",
    options=['無', 'iqr', 'zscore', 'modified_zscore', 'isolation_forest', 'lof', 'stl_residual'],
    index=0,
    help="選擇用於檢測異常值的方法。'無'表示不進行異常值檢測。"
)

outlier_params = {}
if outlier_method == 'iqr':
    outlier_params['iqr_multiplier'] = st.sidebar.slider("IQR 倍數:", min_value=1.0, max_value=5.0, value=1.5, step=0.1, help="IQR 方法中用於定義異常值的倍數。")
elif outlier_method in ['zscore', 'modified_zscore', 'stl_residual']:
    outlier_params['z_threshold'] = st.sidebar.slider("Z-score 閾值:", min_value=1.0, max_value=5.0, value=3.0, step=0.1, help="Z-score / Modified Z-score / STL殘差方法中用於定義異常值的閾值。")
elif outlier_method == 'isolation_forest':
    outlier_params['if_contamination'] = st.sidebar.number_input("Isolation Forest 污染度:", min_value=0.01, max_value=0.5, value=0.1, step=0.01, help="Isolation Forest 中預期異常值的比例。")
elif outlier_method == 'lof':
    outlier_params['n_neighbors'] = st.sidebar.slider("LOF 鄰居數:", min_value=5, max_value=50, value=20, step=1, help="LOF 方法中用於計算局部密度的鄰居數。")


outlier_strategy = '無'
if outlier_method != '無':
    outlier_strategy = st.sidebar.selectbox(
        "異常值處理策略:",
        options=['remove', 'interpolate', 'cap', 'mark'], 
        index=1, 
        help="選擇如何處理檢測到的異常值。'mark'只標記但不修改數據。"
    )


apply_smoothing = st.sidebar.checkbox("應用數據平滑", value=False, key='pages_8_apply_smoothing')
smoothing_window = 1

if apply_smoothing:
    smoothing_window = st.sidebar.slider("平滑處理 (移動平均視窗):", min_value=1, max_value=24, value=3, step=1,
                                         help="移動平均視窗大小（單位與預測頻次相同）。1 表示不進行平滑處理。數值越大，數據越平滑，但可能丟失細節。")

apply_bollinger_bands = st.sidebar.checkbox("顯示布林帶", value=False, key='pages_8_bollinger')
if apply_bollinger_bands:
    bb_window = st.sidebar.slider("布林帶窗口 (移動平均):", min_value=5, max_value=60, value=20, step=1, key='pages_8_bb_window')
    bb_num_std = st.sidebar.slider("布林帶標準差倍數:", min_value=1.0, max_value=3.0, value=2.0, step=0.1, key='pages_8_bb_std')


# --- 模型選擇 ---
st.sidebar.markdown("---")
st.sidebar.subheader("模型選擇與參數")
model_options = []
if prophet_available:
    model_options.append("Prophet")
if statsmodels_available:
    model_options.extend(["SARIMA", "ETS"])

if not model_options:
    st.sidebar.warning("沒有可用的時間序列模型。請檢查安裝提示。")
    st.stop()

selected_model = st.sidebar.selectbox("選擇預測模型:", model_options, key='pages_8_model_select')

# --- 模型選擇指引 ---
if selected_model == "Prophet":
    st.sidebar.info(
        "**Prophet**: 適用於具有明顯季節性（每日、每週、每年）和趨勢變化的數據，尤其在數據缺失或異常值較多時表現良好。由 Facebook 開發。"
    )
elif selected_model == "SARIMA":
    st.sidebar.info(
        "**SARIMA (Seasonal ARIMA)**: 一種經典的統計模型，適用於具有季節性趨勢的數據。需要對數據的自相關和偏自相關特性有一定的了解來手動選擇參數，或使用 Auto-ARIMA 自動搜尋。對缺失值敏感。"
    )
elif selected_model == "ETS":
    st.sidebar.info(
        "**ETS (Exponential Smoothing)**: 透過對誤差 (Error)、趨勢 (Trend) 和季節性 (Seasonality) 組件進行指數平滑來預測。適用於數據模式較穩定，季節性或趨勢明確的場景。對缺失值敏感。"
    )


# --- 定義 sarima_s_options 在模型參數設定之前 ---
# 統一使用小寫頻率代碼
SARIMA_S_OPTIONS_MAP = {
    "小時 (h)": 24, # 每天24小時
    "天 (D)": 7,    # 每週7天
    "週 (W)": 52,    # 每年52週
    "月 (M)": 12,    # 每年12月
    "年 (Y)": 1      # 年度數據，季節性週期通常為1 (趨勢為主)
}

# --- 模型特定參數 ---
if selected_model == "Prophet":
    st.sidebar.caption("Prophet 模型參數")
    prophet_seasonality_mode = st.sidebar.selectbox(
        "季節性模式:", ["additive", "multiplicative"], key='pages_8_prophet_seasonality_mode',
        help="additive (加性): 季節性成分獨立於趨勢。multiplicative (乘性): 季節性成分隨趨勢變化。"
    )
    prophet_changepoint_prior_scale = st.sidebar.slider(
        "趨勢變點彈性 (changepoint_prior_scale):", 0.01, 1.0, 0.05, 0.01, key='pages_8_prophet_changepoint_prior_scale',
        help="越大表示趨勢越靈活，越容易過擬合。越小表示趨勢越平滑。"
    )
    prophet_seasonality_prior_scale = st.sidebar.slider(
        "季節性彈性 (seasonality_prior_scale):", 0.01, 10.0, 1.0, 0.01, key='pages_8_prophet_seasonality_prior_scale',
        help="越大表示季節性成分越靈活。越小表示季節性成分越平滑。"
    )
    prophet_holidays = st.sidebar.checkbox("考慮節假日影響", value=False, key='pages_8_prophet_holidays')

    # Prophet 自動調優選項
    if prophet_available:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Prophet 進階調優")
        auto_tune_prophet = st.sidebar.checkbox(
            "自動調優 Prophet 參數 (實驗性)",
            value=False,
            key='pages_8_auto_tune_prophet',
            help="自動搜索 'changepoint_prior_scale' 和 'seasonality_prior_scale' 的最佳組合。可能耗時較長。"
        )

        if auto_tune_prophet:
            st.sidebar.info("已啟用自動調優，上方手動設定將被忽略。")
            st.sidebar.markdown("**調優範圍設定:**")
            prophet_cps_min = st.sidebar.slider("changepoint_prior_scale 最小值", 0.001, 0.1, 0.01, 0.001, key='prophet_cps_min')
            prophet_cps_max = st.sidebar.slider("changepoint_prior_scale 最大值", 0.1, 1.0, 0.5, 0.01, key='prophet_cps_max')
            prophet_cps_steps = st.sidebar.slider("changepoint_prior_scale 步數", 2, 10, 3, key='prophet_cps_steps')

            prophet_sps_min = st.sidebar.slider("seasonality_prior_scale 最小值", 0.01, 1.0, 0.1, 0.01, key='prophet_sps_min')
            prophet_sps_max = st.sidebar.slider("seasonality_prior_scale 最大值", 1.0, 10.0, 5.0, 0.1, key='prophet_sps_max')
            prophet_sps_steps = st.sidebar.slider("seasonality_prior_scale 步數", 2, 10, 3, key='prophet_sps_steps')

            st.sidebar.markdown("**交叉驗證設定:**")
            initial_period = st.sidebar.number_input(
                "交叉驗證初始訓練數據量 (天):", min_value=30, max_value=365*3, value=180, step=30,
                help="Prophet 交叉驗證的初始訓練數據量，單位為天。"
            )
            period_cv = st.sidebar.number_input(
                "交叉驗證步長 (天):", min_value=7, max_value=365, value=30, step=7,
                help="每次預測之間的間隔，單位為天。"
            )
            horizon_cv = st.sidebar.number_input(
                "交叉驗證預測展望期 (天):", min_value=1, max_value=365, value=30, step=1,
                help="每次交叉驗證預測的未來天數。"
            )
        else:
            prophet_cps_min, prophet_cps_max, prophet_cps_steps = None, None, None
            prophet_sps_min, prophet_sps_max, prophet_sps_steps = None, None, None
            initial_period, period_cv, horizon_cv = None, None, None


elif selected_model == "SARIMA":
    st.sidebar.caption("SARIMA 模型參數 (p,d,q)(P,D,Q,s)")
    
    # --- 新增：自動調優選項 ---
    auto_tune_sarima = st.sidebar.checkbox(
        "自動調優 SARIMA 參數 (Auto-ARIMA)",
        value=False,
        key='pages_8_auto_tune_sarima',
        help="使用 pmdarima 庫自動搜索最佳 (p,d,q)(P,D,Q,s) 參數。此過程可能需要一些時間。"
    )

    if auto_tune_sarima:
        st.sidebar.info("已啟用 Auto-ARIMA，下方手動參數設定將會被忽略。")
        sarima_p, sarima_d, sarima_q, sarima_P, sarima_D, sarima_Q = 0,0,0,0,0,0 
        st.sidebar.markdown(
            "Auto-ARIMA 將根據數據和預測頻次自動選擇最佳的 `(p,d,q)(P,D,Q,s)` 參數。"
        )
    else:
        col1, col2, col3 = st.sidebar.columns(3)
        with col1:
            sarima_p = st.number_input("p:", min_value=0, max_value=5, value=1, step=1, key='pages_8_sarima_p')
            sarima_P = st.number_input("P:", min_value=0, max_value=2, value=1, step=1, key='pages_8_sarima_P')
        with col2:
            sarima_d = st.number_input("d:", min_value=0, max_value=2, value=1, step=1, key='pages_8_sarima_d')
            sarima_D = st.number_input("D:", min_value=0, max_value=1, value=0, step=1, key='pages_8_sarima_D')
        with col3:
            sarima_q = st.number_input("q:", min_value=0, max_value=5, value=0, step=1, key='pages_8_sarima_q')
            sarima_Q = st.number_input("Q:", min_value=0, max_value=2, value=0, step=1, key='pages_8_sarima_Q')
    
    sarima_s = SARIMA_S_OPTIONS_MAP.get(selected_prediction_freq_display, 0)
    if sarima_s == 0:
        st.sidebar.warning("當前預測頻次無法自動設定 SARIMA 季節性週期 (s)。請確保選擇 '小時', '天', '週', '月' 或 '年'。")
        st.sidebar.info("如果數據無明顯季節性或您不希望使用季節性模型，請將 D, Q 設為 0。")
    else:
        st.sidebar.info(f"SARIMA 季節性週期 (s) 將根據預測頻次自動設為: {sarima_s}")
    
elif selected_model == "ETS":
    st.sidebar.caption("ETS (Exponential Smoothing) 模型參數")
    ets_error = st.sidebar.selectbox("誤差 (error):", ['add', 'mul'], key='pages_8_ets_error')
    ets_trend = st.sidebar.selectbox("趨勢 (trend):", ['add', 'mul', None], key='pages_8_ets_trend')
    ets_seasonal = st.sidebar.selectbox("季節性 (seasonal):", ['add', 'mul', None], key='pages_8_ets_seasonal')
    ets_seasonal_periods = st.sidebar.number_input(
        "季節性週期 (seasonal_periods):",
        min_value=1,
        max_value=365,
        value=SARIMA_S_OPTIONS_MAP.get(selected_prediction_freq_display, 1), 
        step=1,
        key='pages_8_ets_seasonal_periods'
    )
    if ets_seasonal is None:
        ets_seasonal_periods = 1 
    
    # ETS 自動調優選項
    if statsmodels_available:
        st.sidebar.markdown("---")
        st.sidebar.subheader("ETS 進階調優")
        auto_tune_ets = st.sidebar.checkbox(
            "自動調優 ETS 參數 (實驗性)",
            value=False,
            key='pages_8_auto_tune_ets',
            help="自動搜索 'error', 'trend', 'seasonal' 的最佳組合。可能耗時較長。"
        )

        if auto_tune_ets:
            st.sidebar.info("已啟用自動調優，上方手動設定將被忽略。")
            ets_error_options_auto = st.sidebar.multiselect("Error 模式 (自動調優):", ['add', 'mul'], default=['add', 'mul'], key='ets_error_auto')
            ets_trend_options_auto = st.sidebar.multiselect("Trend 模式 (自動調優):", ['add', 'mul', None], default=['add', 'mul', None], key='ets_trend_auto')
            ets_seasonal_options_auto = st.sidebar.multiselect("Seasonal 模式 (自動調優):", ['add', 'mul', None], default=['add', 'mul', None], key='ets_seasonal_auto')
            
            st.sidebar.info(f"自動調優時，季節性週期將根據預測頻次自動設為: {SARIMA_S_OPTIONS_MAP.get(selected_prediction_freq_display, 1)}")
        else:
            ets_error_options_auto, ets_trend_options_auto, ets_seasonal_options_auto = None, None, None


# --- 執行預測按鈕 ---
if st.sidebar.button("🔮 執行預測"):
    if (selected_model == "Prophet" and not prophet_available) or \
       ((selected_model == "SARIMA" or selected_model == "ETS") and not statsmodels_available):
        st.error(f"所選模型 ({selected_model}) 的必要庫未安裝或無法載入。請參考錯誤提示安裝。")
        st.stop()
    
    if selected_model == "SARIMA" and auto_tune_sarima and not statsmodels_available: 
        st.error("SARIMA 自動調優功能需要 pmdarima 庫，但它未安裝或無法載入。請參考錯誤提示安裝。")
        st.stop()

    # --- 異常值處理前置檢查，確保相關庫可用 ---
    if outlier_method != '無':
        try:
            if outlier_method == 'isolation_forest':
                from sklearn.ensemble import IsolationForest
            elif outlier_method == 'lof':
                from sklearn.neighbors import LocalOutlierFactor
            elif outlier_method in ['zscore', 'modified_zscore']:
                from scipy import stats
            elif outlier_method == 'stl_residual':
                from statsmodels.tsa.seasonal import STL
        except ImportError as e:
            st.error(f"錯誤：您選擇的異常值檢測方法 '{outlier_method}' 需要額外的庫，但其未安裝或無法載入：`{e}`。")
            st.info("請運行 `pip install scikit-learn scipy statsmodels` 安裝缺失的庫。")
            st.stop()


    df_loaded = load_data(selected_station, st.session_state.get('parameter_info', {})) # 載入原始數據

    selected_station_name = get_station_name_from_id(selected_station)

    if df_loaded.empty or selected_param_col not in df_loaded.columns:
        if df_loaded.empty:
            st.error(f"所選測站 '{selected_station_name}' 沒有成功載入任何數據。")
        else:
            st.error(f"所選測站 '{selected_station_name}' 的數據文件缺少參數 '{selected_param_display_original}' (原始列名: '{selected_param_col}')。")
            st.info(f"數據中可用的列: {df_loaded.columns.tolist()}")
        st.stop()

    st.info(f"正在對測站 **{selected_station_name}** 的參數 **{selected_param_display_original}** 執行 {selected_model} 預測...")

    # --- 數據預處理 (開始) ---
    df_processed = df_loaded[['ds', selected_param_col]].copy()
    df_processed.columns = ['ds', 'y']

    # `pd.to_datetime` 在 `load_data` 中已經處理過，這裡再次檢查並排序
    df_processed['ds'] = pd.to_datetime(df_processed['ds'], errors='coerce') 
    df_processed.sort_values('ds', inplace=True)
    df_processed.dropna(subset=['ds'], inplace=True) 

    train_start_datetime = pd.to_datetime(train_start_date)
    train_end_datetime = pd.to_datetime(train_end_date) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)

    df_processed = df_processed[
        (df_processed['ds'] >= train_start_datetime) &
        (df_processed['ds'] <= train_end_datetime)
    ].copy()

    if df_processed.empty:
        st.error(f"在選定的訓練時間範圍 ({train_start_date} 至 {train_end_date}) 內沒有找到數據。請調整時間範圍。")
        st.stop()
    
    if df_processed['ds'].duplicated().any():
        st.warning("警告：訓練數據中存在重複的時間戳，可能影響模型訓練。將移除重複項。")
        df_processed.drop_duplicates(subset=['ds'], keep='first', inplace=True)

    # 在重採樣前保存一份用於可視化原始數據的副本
    df_original_for_plot_raw = df_processed.copy()


    # 對 'ds' 進行重採樣
    df_processed = df_processed.set_index('ds').resample(selected_freq_pandas.upper()).mean().reset_index()
    df_original_for_plot_raw = df_original_for_plot_raw.set_index('ds').resample(selected_freq_pandas.upper()).mean().reset_index()
    df_original_for_plot = df_original_for_plot_raw.copy()


    # --- 缺失值處理 ---
    if missing_value_strategy == '前向填充 (ffill)':
        df_processed['y'] = df_processed['y'].ffill()
    elif missing_value_strategy == '後向填充 (bfill)':
        df_processed['y'] = df_processed['y'].bfill()
    elif missing_value_strategy == '線性插值 (interpolate)':
        df_processed['y'] = df_processed['y'].interpolate(method='linear')
    elif missing_value_strategy == '移除缺失值 (dropna)':
        df_processed = df_processed.dropna(subset=['y'])
    elif missing_value_strategy == '不處理 (保持原樣)':
        st.info("已選擇不處理缺失值。請注意，某些模型可能無法處理 NaN 值。")
    
    # 針對缺失值處理後數據是否為空進行檢查
    if df_processed['y'].isnull().all():
        st.error(f"在經過預處理後，參數 '{selected_param_display}' 的數據全部為缺失值。無法進行預測。")
        st.stop()
        
    # 如果選擇不處理，並且數據中仍有 NaN，則發出警告
    if missing_value_strategy == '不處理 (保持原樣)' and df_processed['y'].isnull().any():
        st.warning(f"警告：您選擇了不處理缺失值，數據中仍包含 {df_processed['y'].isnull().sum()} 個缺失值。SARIMA 或 ETS 模型可能因此失敗。")

    # --- 執行異常值檢測與處理 ---
    is_outlier_series_original_detection = pd.Series(False, index=df_processed.index) 
    num_outliers = 0 
    if outlier_method != '無':
        st.info(f"正在執行異常值檢測 (方法: {outlier_method}) 和處理 (策略: {outlier_strategy})...")
        
        stl_s_period_for_detection = SARIMA_S_OPTIONS_MAP.get(selected_prediction_freq_display, None)
        
        is_outlier_series_original_detection = detect_outliers(
            df_processed, 
            'y',
            method=outlier_method,
            iqr_multiplier=outlier_params.get('iqr_multiplier', 1.5),
            z_threshold=outlier_params.get('z_threshold', 3.0),
            if_contamination=outlier_params.get('if_contamination', 'auto'),
            n_neighbors=outlier_params.get('n_neighbors', 20),
            stl_seasonal_period_input=stl_s_period_for_detection, 
            selected_freq_pandas_input=selected_freq_pandas.lower() 
        )
        
        num_outliers = is_outlier_series_original_detection.sum()
        if num_outliers > 0:
            st.info(f"檢測到 {num_outliers} 個異常值。")
            
            df_processed = handle_outliers(df_processed, 'y', is_outlier_series_original_detection, strategy=outlier_strategy)
        else:
            st.info("未檢測到異常值。")
            
    df_original_for_plot['is_outlier_original_detection'] = is_outlier_series_original_detection.reindex(df_original_for_plot.index, fill_value=False)
    df_original_for_plot.loc[df_original_for_plot['y'].isnull(), 'is_outlier_original_detection'] = False


    # --- 數據平滑 (在異常值處理之後) ---
    if apply_smoothing and smoothing_window > 1:
        df_processed['y_smoothed'] = df_processed['y'].rolling(window=smoothing_window, min_periods=1, center=True).mean()
        df_processed['y'] = df_processed['y_smoothed']
        st.info(f"數據已應用移動平均平滑處理，窗口大小為 {smoothing_window}。")

    # 最終檢查數據是否可用於模型訓練
    if selected_model != "Prophet" and df_processed['y'].isnull().any():
        st.error(f"錯誤：所選模型 **{selected_model}** 不支持缺失值，但數據中仍包含缺失值。請在左側側邊欄選擇其他缺失值處理策略，例如 '移除缺失值' 或 '線性插值'。")
        st.stop()
    elif selected_model == "Prophet":
        df_processed.dropna(subset=['ds', 'y'], inplace=True) 

    if df_processed.empty or df_processed['y'].count() < 2: 
        st.error("經過數據預處理和時間範圍篩選後，沒有足夠的有效數據用於預測。請檢查原始數據、時間範圍和預處理選項。")
        st.stop()
        
    # --- 數據概覽 ---
    st.subheader("📊 數據概覽與品質分析")
    total_duration_td = df_processed['ds'].max() - df_processed['ds'].min()
    total_duration_days = total_duration_td.total_seconds() / (24*3600) 
    st.write(f"**使用數據區間**: 從 **{df_processed['ds'].min().strftime('%Y-%m-%d %H:%M')}** 到 **{df_processed['ds'].max().strftime('%Y-%m-%d %H:%M')}**")
    st.write(f"**總時長**: **{total_duration_td}** (約 {total_duration_days:.2f} 天)") 
    st.write(f"**總筆數**: **{len(df_processed)}** 筆")
    try:
        inferred_freq = pd.infer_freq(df_processed['ds'])
    except ValueError:
        inferred_freq = '無法精確推斷 (數據可能間隔不一致)'
    st.write(f"**數據頻次 (預處理後)**: **{selected_freq_pandas.lower()}** (原始推斷: **{inferred_freq}**)") 
    
    df_for_quality_check = df_processed[['ds', 'y']].set_index('ds').rename(columns={'y': selected_param_col}).copy()
    
    quality_report = analyze_data_quality(df_for_quality_check, relevant_params=[selected_param_col])

    metrics = {} 
    if selected_param_col in quality_report:
        metrics = quality_report[selected_param_col]
        st.write(f"**參數: {selected_param_display_original}**")
        st.write(f"- 總記錄數: {metrics.get('total_records', 'N/A')}")
        st.write(f"- 有效記錄數: {metrics.get('valid_count', 'N/A')}")
        st.write(f"- 缺失值數量: {metrics.get('missing_count', 'N/A')} (**{metrics.get('missing_percentage', 0):.2f}%**)")
        if metrics.get('is_numeric', True):
            st.write(f"- 零值數量: {metrics.get('zero_count', 'N/A')}")
            st.write(f"- 負值數量: {metrics.get('negative_count', 'N/A')}")
            st.write(f"- 潛在 IQR 異常值數量: {metrics.get('outlier_iqr_count', 'N/A')}")

            quality_data = {
                '類型': ['有效值', '缺失值', '零值', '負值', '潛在異常值'],
                '數量': [
                    metrics.get('valid_count', 0),
                    metrics.get('missing_count', 0),
                    metrics.get('zero_count', 0),
                    metrics.get('negative_count', 0),
                    metrics.get('outlier_iqr_count', 0)
                ]
            }
            quality_df = pd.DataFrame(quality_data)
            quality_df = quality_df[quality_df['數量'] > 0] 

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
                
                fig_quality.update_layout(showlegend=True)
                
                st.plotly_chart(fig_quality, use_container_width=True)
            else:
                st.info("數據品質非常高，沒有缺失值、零值、負值或異常值。")
        else:
            st.warning(f"無法為參數 '{selected_param_display_original}' 生成數據品質報告。")


    # --- 模型訓練與預測 ---
    forecast = pd.DataFrame()
    m = None
    
    # 儲存自動選擇的 SARIMA/Prophet/ETS 參數
    auto_params_display = ""
    model_summary_text = ""

    with st.spinner(f"正在訓練 {selected_model} 模型並生成預測..."):
        try:
            if selected_model == "Prophet":
                # Prophet 要求 ds 為 datetime，y 為數值
                df_processed_prophet = df_processed[['ds', 'y']].copy().dropna()
                if len(df_processed_prophet) < 2:
                    st.error("Prophet 模型需要至少兩個有效數據點才能訓練。請擴大訓練數據範圍或調整數據頻率。")
                    st.stop()

                if auto_tune_prophet and prophet_available:
                    st.info("正在執行 Prophet 參數自動調優，這可能需要一些時間... (基於交叉驗證)")
                    
                    param_grid = {
                        'changepoint_prior_scale': np.linspace(prophet_cps_min, prophet_cps_max, prophet_cps_steps).tolist(),
                        'seasonality_prior_scale': np.linspace(prophet_sps_min, prophet_sps_max, prophet_sps_steps).tolist(),
                    }
                    grid = list(ParameterGrid(param_grid))

                    best_params = None
                    best_rmse = float('inf')
                    
                    # 關閉 Prophet 的日誌，避免大量輸出
                    prophet_logger.setLevel(logging.WARNING)

                    progress_text = "Prophet 自動調優進度: {current_param_idx}/{total_params} 組合已測試。"
                    progress_bar = st.progress(0, text=progress_text.format(current_param_idx=0, total_params=len(grid)))

                    for i, params in enumerate(grid):
                        with st.spinner(f"正在測試 Prophet 參數組合 {i+1}/{len(grid)}: {params}"):
                            m_cv = Prophet(
                                seasonality_mode=prophet_seasonality_mode,
                                changepoint_prior_scale=params['changepoint_prior_scale'],
                                seasonality_prior_scale=params['seasonality_prior_scale']
                            )
                            if prophet_holidays:
                                m_cv.add_country_holidays(country_name='TW')
                            
                            try:
                                m_cv.fit(df_processed_prophet)
                                
                                # 將 initial, period, horizon 從天數轉換為頻率單位
                                # 注意：Prophet 的 initial/period/horizon 參數可以直接接受 'Xd' 格式
                                # 這裡的轉換主要是為了檢查數據量是否足夠
                                # 這裡只需要確保 df_processed_prophet 有足夠的長度
                                if len(df_processed_prophet) < initial_period + horizon_cv:
                                    st.warning(f"數據量 ({len(df_processed_prophet)}) 不足 ({initial_period} 天初始數據 + {horizon_cv} 天預測展望期)。跳過 Prophet 交叉驗證。")
                                    rmse_current = float('inf')
                                else:
                                    df_cv = cross_validation(
                                        m_cv,
                                        initial=f'{initial_period} days',
                                        period=f'{period_cv} days',
                                        horizon=f'{horizon_cv} days',
                                        parallel="processes"
                                    )
                                    if df_cv.empty:
                                        rmse_current = float('inf')
                                    else:
                                        df_p = performance_metrics(df_cv)
                                        rmse_current = df_p['rmse'].mean()

                                if rmse_current < best_rmse:
                                    best_rmse = rmse_current
                                    best_params = params
                            except Exception as e:
                                # st.warning(f"Prophet 參數組合 {params} 訓練或交叉驗證失敗：{e}")
                                pass # 繼續下一個組合，不中斷 Streamlit

                        progress_bar.progress((i + 1) / len(grid), text=progress_text.format(current_param_idx=i+1, total_params=len(grid)))

                    prophet_logger.setLevel(logging.INFO) # 恢復日誌級別

                    if best_params:
                        st.success(f"Prophet 自動調優完成。最佳參數為: {best_params}")
                        prophet_changepoint_prior_scale = best_params['changepoint_prior_scale']
                        prophet_seasonality_prior_scale = best_params['seasonality_prior_scale']
                        auto_params_display = f"changepoint_prior_scale={prophet_changepoint_prior_scale:.4f}, seasonality_prior_scale={prophet_seasonality_prior_scale:.4f}"
                    else:
                        st.warning("Prophet 自動調優未能找到最佳參數，將使用手動設定或預設值。")
                        auto_params_display = "自動調優失敗或數據不足，使用手動設定值。"
                else:
                    auto_params_display = "手動設定"

                # 使用手動設定或自動調優後的參數訓練最終模型
                m = Prophet(
                    seasonality_mode=prophet_seasonality_mode,
                    changepoint_prior_scale=prophet_changepoint_prior_scale,
                    seasonality_prior_scale=prophet_seasonality_prior_scale
                )
                if prophet_holidays:
                    m.add_country_holidays(country_name='TW')
                    st.info("已為 Prophet 模型添加台灣節假日。")

                m.fit(df_processed_prophet) # 使用清理後的數據進行訓練
                # --- 修正點 2: 統一使用小寫頻率符號 ---
                future = m.make_future_dataframe(periods=forecast_period_value, freq=selected_freq_pandas.lower())
                forecast = m.predict(future)

            elif selected_model == "SARIMA":
                # --- 修正點 2: 統一使用小寫頻率符號 ---
                sarima_s_actual = SARIMA_S_OPTIONS_MAP.get(selected_prediction_freq_display, 1) 

                if auto_tune_sarima:
                    st.info("正在執行 Auto-ARIMA 搜索最佳參數，這可能需要一些時間...")
                    
                    model_auto_arima = pm.auto_arima(df_processed['y'].dropna(), 
                                                     seasonal=True if sarima_s_actual > 1 else False,
                                                     m=sarima_s_actual if sarima_s_actual > 1 else 1, 
                                                     D=sarima_D if not auto_tune_sarima else None, 
                                                     max_p=5, max_d=2, max_q=5,
                                                     max_P=2, max_D=1, max_Q=2,
                                                     start_p=0, start_q=0, start_P=0, start_Q=0,
                                                     trace=False, 
                                                     error_action='ignore',
                                                     suppress_warnings=True,
                                                     stepwise=True,
                                                     n_fits=50 
                                                    )
                    
                    sarima_p, sarima_d, sarima_q = model_auto_arima.order
                    sarima_P, sarima_D, sarima_Q, _ = model_auto_arima.seasonal_order
                    sarima_s = sarima_s_actual 
                    auto_params_display = f"(p={sarima_p}, d={sarima_d}, q={sarima_q})(P={sarima_P}, D={sarima_D}, Q={sarima_Q}, s={sarima_s})"
                    st.success(f"Auto-ARIMA 選定的最佳參數為: {auto_params_display}")
                else:
                    sarima_s = sarima_s_actual 
                    auto_params_display = f"(p={sarima_p}, d={sarima_d}, q={sarima_q})(P={sarima_P}, D={sarima_D}, Q={sarima_Q}, s={sarima_s})"

                min_sarima_data_points = max(2 * sarima_s, 2 * (sarima_p + sarima_P)) if sarima_s > 1 else (sarima_p + sarima_d + sarima_q + 10)
                if len(df_processed) < min_sarima_data_points:
                    st.error(f"數據點 ({len(df_processed)}) 不足，無法訓練 SARIMA 模型。至少需要約 {min_sarima_data_points} 點。請調整數據範圍或預測頻次。")
                    st.stop()

                model_sarima = SARIMAX(df_processed['y'],
                                       order=(sarima_p, sarima_d, sarima_q),
                                       seasonal_order=(sarima_P, sarima_D, sarima_Q, sarima_s) if sarima_s > 1 else (0,0,0,0),
                                       enforce_stationarity=False,
                                       enforce_invertibility=False)
                results_sarima = model_sarima.fit(disp=False)
                model_summary_text = results_sarima.summary().as_text()
                
                n_predict = len(df_processed) + forecast_period_value
                sarima_forecast = results_sarima.get_prediction(start=0, end=n_predict - 1)
                
                forecast_mean = sarima_forecast.predicted_mean
                conf_int = sarima_forecast.conf_int(alpha=0.05)

                forecast = pd.DataFrame({
                    'ds': pd.date_range(start=df_processed['ds'].min(), periods=n_predict, freq=selected_freq_pandas.lower()), 
                    'yhat': forecast_mean,
                    'yhat_lower': conf_int.iloc[:, 0],
                    'yhat_upper': conf_int.iloc[:, 1]
                })
                forecast = forecast[forecast['ds'].isin(df_processed['ds']) | (forecast['ds'] > df_processed['ds'].max())].copy()


            elif selected_model == "ETS":
                ets_seasonal_periods_actual = SARIMA_S_OPTIONS_MAP.get(selected_prediction_freq_display, 1)

                if auto_tune_ets and statsmodels_available:
                    st.info("正在執行 ETS 參數自動調優，這可能需要一些時間...")
                    
                    param_grid = {
                        'error': ets_error_options_auto,
                        'trend': ets_trend_options_auto,
                        'seasonal': ets_seasonal_options_auto,
                    }
                    grid = list(ParameterGrid(param_grid))

                    best_params = None
                    best_rmse = float('inf')
                    
                    progress_text = "ETS 自動調優進度: {current_param_idx}/{total_params} 組合已測試。"
                    progress_bar = st.progress(0, text=progress_text.format(current_param_idx=0, total_params=len(grid)))

                    for i, params in enumerate(grid):
                        with st.spinner(f"正在測試 ETS 參數組合 {i+1}/{len(grid)}: {params}"):
                            current_ets_seasonal_periods = ets_seasonal_periods_actual if params['seasonal'] else 1

                            try:
                                min_ets_data_points = current_ets_seasonal_periods * 2 if params['seasonal'] else 10
                                if len(df_processed) < min_ets_data_points:
                                    rmse_current = float('inf')
                                else:
                                    model_ets_cv = ExponentialSmoothing(
                                        df_processed['y'],
                                        seasonal_periods=current_ets_seasonal_periods if params['seasonal'] else None,
                                        trend=params['trend'],
                                        seasonal=params['seasonal'],
                                        initialization_method="estimated"
                                    ).fit(disp=False)

                                    y_pred_cv = model_ets_cv.fittedvalues
                                    valid_indices_cv = ~np.isnan(df_processed['y']) & ~np.isnan(y_pred_cv)
                                    if valid_indices_cv.any():
                                        rmse_current = np.sqrt(mean_squared_error(df_processed['y'][valid_indices_cv], y_pred_cv[valid_indices_cv]))
                                    else:
                                        rmse_current = float('inf')

                                if rmse_current < best_rmse:
                                    best_rmse = rmse_current
                                    best_params = params
                                    best_params['seasonal_periods'] = current_ets_seasonal_periods
                            except Exception as e:
                                pass 

                        progress_bar.progress((i + 1) / len(grid), text=progress_text.format(current_param_idx=i+1, total_params=len(grid)))

                    if best_params:
                        st.success(f"ETS 自動調優完成。最佳參數為: {best_params}")
                        ets_error = best_params['error']
                        ets_trend = best_params['trend']
                        ets_seasonal = best_params['seasonal']
                        ets_seasonal_periods = best_params['seasonal_periods']
                        auto_params_display = f"error='{ets_error}', trend='{ets_trend}', seasonal='{ets_seasonal}', seasonal_periods={ets_seasonal_periods}"
                    else:
                        st.warning("ETS 自動調優未能找到最佳參數，將使用手動設定或預設值。")
                        auto_params_display = "自動調優失敗或數據不足，使用手動設定值。"
                else:
                    ets_seasonal_periods = ets_seasonal_periods_actual
                    auto_params_display = "手動設定"

                min_ets_data_points = ets_seasonal_periods * 2 if ets_seasonal else 10
                if len(df_processed) < min_ets_data_points:
                    st.error(f"數據點 ({len(df_processed)}) 不足，無法訓練 ETS 模型。至少需要約 {min_ets_data_points} 點。請調整數據範圍或預測頻次。")
                    st.stop()

                model_ets = ExponentialSmoothing(
                    df_processed['y'],
                    seasonal_periods=ets_seasonal_periods if ets_seasonal else None,
                    trend=ets_trend,
                    seasonal=ets_seasonal,
                    initialization_method="estimated"
                ).fit()
                
                model_summary_text = model_ets.summary().as_text()

                ets_forecast = model_ets.predict(start=0, end=len(df_processed) + forecast_period_value -1)

                forecast = pd.DataFrame({
                    'ds': pd.date_range(start=df_processed['ds'].min(), periods=len(df_processed) + forecast_period_value, freq=selected_freq_pandas.lower()), 
                    'yhat': ets_forecast
                })
                forecast['yhat_lower'] = np.nan
                forecast['yhat_upper'] = np.nan
                forecast = forecast[forecast['ds'].isin(df_processed['ds']) | (forecast['ds'] > df_processed['ds'].max())].copy()

        except Exception as e:
            st.error(f"模型訓練或預測失敗：{e}。請檢查數據或調整模型參數。")
            st.stop()

    st.success(f"{selected_model} 模型預測完成！")

    # --- 性能指標計算與顯示 (針對主預測) ---
    st.subheader("📊 模型性能指標 (訓練數據)")
    if auto_params_display:
        st.info(f"**{selected_model} 使用參數**: {auto_params_display}")

    actual_vs_predicted = pd.merge(df_processed[['ds', 'y']], forecast[['ds', 'yhat']], on='ds', how='inner')
    
    performance_metrics_report = {} 

    if not actual_vs_predicted.empty:
        y_true = actual_vs_predicted['y']
        y_pred = actual_vs_predicted['yhat']

        valid_indices = ~np.isnan(y_true) & ~np.isnan(y_pred)
        y_true_clean = y_true[valid_indices]
        y_pred_clean = y_pred[valid_indices]

        if len(y_true_clean) > 0:
            rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
            mae = mean_absolute_error(y_true_clean, y_pred_clean)
            
            correlation = np.nan
            if len(y_true_clean) > 1 and len(y_pred_clean) > 1 and y_true_clean.std() > 0 and y_pred_clean.std() > 0:
                try:
                    correlation, _ = pearsonr(y_true_clean, y_pred_clean)
                except Exception as e:
                    st.warning(f"計算相關係數失敗: {e}")
            else:
                st.warning("數據點不足或數據無變化，無法計算相關係數。")

            mape = np.nan
            if (y_true_clean != 0).any():
                non_zero_mape_indices = y_true_clean != 0
                if non_zero_mape_indices.any():
                    mape = np.mean(np.abs((y_true_clean[non_zero_mape_indices] - y_pred_clean[non_zero_mape_indices]) / y_true_clean[non_zero_mape_indices])) * 100
                else:
                    st.warning("警告: 無法計算 MAPE，因為訓練數據中的實際值全部為零。")
            else:
                st.warning("警告: 無法計算 MAPE，因為訓練數據中的實際值全部為零。")

            st.markdown(f"**均方根誤差 (RMSE):** `{rmse:.4f}`")
            st.markdown(f"**平均絕對誤差 (MAE):** `{mae:.4f}`")
            if not np.isnan(mape):
                st.markdown(f"**平均絕對百分比誤差 (MAPE):** `{mape:.2f}%`")
            else:
                st.markdown("**平均絕對百分比誤差 (MAPE):** `N/A` (數據問題無法計算)")
            
            if not np.isnan(correlation):
                st.markdown(f"**相關係數 (Correlation Coefficient):** `{correlation:.4f}`")
            else:
                st.markdown("**相關係數 (Correlation Coefficient):** `N/A` (數據點不足或計算失敗)")

            performance_metrics_report['RMSE'] = f"{rmse:.4f}"
            performance_metrics_report['MAE'] = f"{mae:.4f}"
            performance_metrics_report['MAPE'] = f"{mape:.2f}%" if not np.isnan(mape) else "N/A"
            performance_metrics_report['Correlation Coefficient'] = f"{correlation:.4f}" if not np.isnan(correlation) else "N/A"

            st.markdown("---")
            st.markdown("這些指標衡量模型在訓練數據上的擬合準確性：")
            st.markdown("- **RMSE (Root Mean Squared Error)**：預測值與實際值偏差的平方平均數的平方根。值越小表示模型越準確。對大誤差的懲罰更大。")
            st.markdown("- **MAE (Mean Absolute Error)**：預測值與實際值偏差的絕對值平均數。值越小表示模型越準確。")
            st.markdown("- **MAPE (Mean Absolute Percentage Error)**：預測誤差的百分比。對於理解誤差相對於實際值的比例很有用。值越小越好。")
            st.markdown("- **相關係數 (Correlation Coefficient)**：衡量兩個變量（實際值和預測值）之間線性關係的強度和方向。值越接近 `1` 表示正向線性關係越強；越接近 `-1` 表示負向線性關係越強；接近 `0` 表示線性關係很弱。理想情況下，此值應接近 `1`。")
        else:
            st.warning("沒有足夠的重疊數據點來計算性能指標。")
    else:
        st.warning("無法將實際值與預測值對齊以計算性能指標。請檢查數據和預測時間範圍。")


    ### 📈 預測結果視覺化

    st.subheader("📈 預測結果")
    if not forecast.empty:
        # 合併原始數據和處理後的數據，用於可視化對比
        plot_df = pd.merge(df_original_for_plot[['ds', 'y', 'is_outlier_original_detection']],
                            df_processed[['ds', 'y']], 
                            on='ds',
                            how='inner', 
                            suffixes=('_original', '_processed'))
        
        # 合併預測結果
        plot_df = pd.merge(plot_df, forecast, on='ds', how='left')


        fig = go.Figure()

        # 添加原始實際數據 (可能含有斷裂點)
        fig.add_trace(go.Scatter(
            x=plot_df['ds'],
            y=plot_df['y_original'],
            mode='lines',
            name='原始數據',
            line=dict(color='blue', dash='dot')
        ))
        
        # 添加經過預處理（包括異常值處理和平滑）的數據
        fig.add_trace(go.Scatter(
            x=plot_df['ds'],
            y=plot_df['y_processed'],
            mode='lines',
            name='處理後數據',
            line=dict(color='darkgreen', width=2)
        ))


        # --- 標記原始數據中的異常值 ---
        outlier_df_to_show = plot_df[plot_df['is_outlier_original_detection'] & plot_df['y_original'].notna()].copy()
        if not outlier_df_to_show.empty:
            fig.add_trace(go.Scatter(
                x=outlier_df_to_show['ds'],
                y=outlier_df_to_show['y_original'],
                mode='markers',
                name='檢測到的原始異常值',
                marker=dict(color='red', size=8, symbol='x', line=dict(width=1, color='DarkRed'))
            ))
        else:
            st.info("未檢測到有效異常值點來顯示。")
        # --- 標記異常值結束 ---

        # 預測線
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat'],
            mode='lines',
            name='預測值',
            line=dict(color='red', dash='dash', width=2)
        ))

        # 預測區間
        if 'yhat_lower' in forecast.columns and not forecast['yhat_lower'].isnull().all():
            fig.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat_lower'],
                mode='lines',
                name='預測下限',
                line=dict(color='lightcoral', width=0),
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat_upper'],
                mode='lines',
                name='預測上限',
                fill='tonexty',
                fillcolor='rgba(255,0,0,0.1)',
                line=dict(color='lightcoral', width=0)
            ))
        
        # 布林帶
        if apply_bollinger_bands:
            if len(df_processed) >= bb_window:
                df_processed_bb = calculate_bollinger_bands(df_processed.copy(), window=bb_window, num_std_dev=bb_num_std)
                if df_processed_bb is not None:
                    fig.add_trace(go.Scatter(
                        x=df_processed_bb['ds'], y=df_processed_bb['MA'], mode='lines', name='布林帶中軌 (MA)',
                        line=dict(color='purple', dash='dot')
                    ))
                    fig.add_trace(go.Scatter(
                        x=df_processed_bb['ds'], y=df_processed_bb['Upper'], mode='lines', name='布林上軌',
                        line=dict(color='green', dash='dot')
                    ))
                    fig.add_trace(go.Scatter(
                        x=df_processed_bb['ds'], y=df_processed_bb['Lower'], mode='lines', name='布林下軌',
                        line=dict(color='orange', dash='dot')
                    ))
            else:
                st.warning(f"數據點 ({len(df_processed)}) 少於布林帶窗口 {bb_window}，無法顯示布林帶。")


        forecast_unit_display = selected_prediction_freq_display.split(' ')[0]
        
        fig.update_layout(
            title=f"{selected_station_name} - {selected_param_display_original} 未來 {forecast_period_value} {forecast_unit_display} 預測 ({selected_model})",
            xaxis_title="時間",
            yaxis_title=f"{selected_param_display_original} {param_unit}",
            hovermode="x unified",
            height=600,
            xaxis=dict(rangeslider_visible=True) 
        )

        st.plotly_chart(fig, use_container_width=True)

        ### 💾 下載預測結果與報告

        st.markdown("您可以下載包含預測值和不確定性區間的 CSV 文件，或者下載**互動式 HTML 圖表**，以及**預測報告**。")

        # --- 下載互動式 HTML 圖表 ---
        if not forecast.empty and fig:
            html_export_string = pio.to_html(fig, full_html=True, include_plotlyjs='cdn')

            st.download_button(
                label="下載互動式 HTML 圖表",
                data=html_export_string.encode('utf-8'),
                file_name=f"{selected_station_name}_{selected_param_col}_{selected_model}_forecast_chart.html",
                mime="text/html",
                help="下載可獨立打開並互動的圖表 HTML 文件"
            )


        download_df = forecast.copy()
        download_df.rename(columns={'ds': '時間', 'yhat': f'預測值_{selected_param_display_original}',
                                     'yhat_lower': f'預測下限_{selected_param_display_original}',
                                     'yhat_upper': f'預測上限_{selected_param_display_original}'}, inplace=True)
        download_df['時間'] = download_df['時間'].dt.strftime('%Y-%m-%d %H:%M:%S')

        csv_data = download_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="下載預測 CSV 文件",
            data=csv_data,
            file_name=f"{selected_station_name}_{selected_param_col}_{selected_model}_forecast.csv",
            mime="text/csv",
        )

        # --- 生成並下載預測報告 ---
        report_content = f"""
# 時間序列預測報告 - {selected_station_name} - {selected_param_display_original}

## 1. 預測概覽
- **測站**: {selected_station_name}
- **預測參數**: {selected_param_display_original} ({param_unit})
- **預測模型**: {selected_model}
- **預測未來時長**: {forecast_period_value} {forecast_unit_display}
- **預測頻次**: {selected_freq_pandas.lower()}

## 2. 數據概覽與品質分析
- **使用數據區間**: 從 {df_processed['ds'].min().strftime('%Y-%m-%d %H:%M')} 到 {df_processed['ds'].max().strftime('%Y-%m-%d %H:%M')}
- **總時長**: {total_duration_td} (約 {total_duration_days:.2f} 天)
- **總筆數**: {len(df_processed)} 筆
- **數據頻次 (預處理後)**: {selected_freq_pandas.lower()}

### 數據品質報告
- 總記錄數: {metrics.get('total_records', 'N/A')}
- 有效記錄數: {metrics.get('valid_count', 'N/A')}
- 缺失值數量: {metrics.get('missing_count', 'N/A')} ({metrics.get('missing_percentage', 0):.2f}%)
- 零值數量: {metrics.get('zero_count', 'N/A')}
- 負值數量: {metrics.get('negative_count', 'N/A')}
- 潛在 IQR 異常值數量: {metrics.get('outlier_iqr_count', 'N/A')}

## 3. 數據預處理設定
- **缺失值處理策略**: {missing_value_strategy}
- **異常值檢測方法**: {outlier_method}
- **異常值處理策略**: {outlier_strategy}
"""
        if outlier_method != '無':
            report_content += f"  - 檢測到的異常值數量: {num_outliers}\n"
        
        report_content += f"""- **數據平滑處理**: {'是' if apply_smoothing and smoothing_window > 1 else '否'}"""
        if apply_smoothing and smoothing_window > 1:
            report_content += f""" (移動平均視窗: {smoothing_window})
"""
        else:
            report_content += "\n"

        report_content += f"""
## 4. 模型參數與訓練
- **模型類型**: {selected_model}
- **使用參數**: {auto_params_display}
"""
        if model_summary_text:
            report_content += f"""
### 模型訓練摘要
{model_summary_text}


"""

        report_content += f"""
## 5. 模型性能指標 (訓練數據)
- **均方根誤差 (RMSE)**: {performance_metrics_report.get('RMSE', 'N/A')}
- **平均絕對誤差 (MAE)**: {performance_metrics_report.get('MAE', 'N/A')}
- **平均絕對百分比誤差 (MAPE)**: {performance_metrics_report.get('MAPE', 'N/A')}
- **相關係數 (Correlation Coefficient)**: {performance_metrics_report.get('Correlation Coefficient', 'N/A')}

---
報告生成時間: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

        st.download_button(
            label="下載預測報告 (TXT)",
            data=report_content.encode('utf-8'),
            file_name=f"{selected_station_name}_{selected_param_col}_{selected_model}_forecast_report.txt",
            mime="text/plain",
            help="下載包含預測設定、數據品質、模型參數和性能指標的文本報告"
        )


    else:
        st.warning("預測結果為空，無法進行可視化或下載。")
