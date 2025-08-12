import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.helpers import get_station_name_from_id, initialize_session_state, load_year_data, convert_df_to_csv, PARAMETER_INFO, analyze_data_quality
import numpy as np
import io
import datetime
import zipfile

# 導入 ruptures 庫，用於趨勢變點偵測
try:
    import ruptures as rpt
    ruptures_available = True
except ImportError:
    ruptures_available = False
    st.warning("警告：ruptures 庫未安裝。趨勢變點偵測功能將無法使用。請運行 `pip install ruptures`。")


initialize_session_state()
st.title("🔬 單站資料探索")
st.write("檢視特定測站在特定年月份的詳細時序資料。")
st.markdown("---")

# 從 session_state 讀取共享資料.get('locations', [])
locations = st.session_state.get('locations', [])
base_data_path = st.session_state.get('base_data_path', '')

available_years = st.session_state.get('available_years', [])
if not available_years:
    st.warning("沒有偵測到任何可用的年份資料，請檢查資料夾設定或返回主頁面重新載入。")
    st.stop()

# --- 表單用於控制主要的數據載入和分析 ---
with st.form("main_dashboard_form_pages2"):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        index = 0
        if st.query_params.get('station'):
            index = locations.index(st.query_params['station']) if st.query_params['station'] in locations else 0

        station = st.selectbox("選擇測站", locations, key='pages_2_ss_station_main', index=index, format_func=get_station_name_from_id)
    with col2:
        default_year_index = len(available_years) - 1 if available_years else 0
        year = st.selectbox("選擇年份", available_years, index=default_year_index, key='pages_2_ss_year')
    with col3:
        month_options = {0: "全年"}
        month_options.update({i: f"{i}月" for i in range(1, 13)})
        month = st.selectbox("選擇月份", list(month_options.keys()), format_func=lambda x: month_options[x], key='pages_2_ss_month')
    with col4:
        chart_type = st.selectbox(
            "選擇圖表類型",
            ('折線圖', '散佈圖 (含趨勢線)', '面積圖', '分佈直方圖'),
            key='pages_2_ss_chart_type'
        )
    submitted = st.form_submit_button("🚀 產生統計報告")



# --- 數據預處理控制項 ---
st.sidebar.markdown("---")
st.sidebar.header("數據預處理")

impute_method = st.sidebar.selectbox(
    "缺失值處理方法:",
    ("不處理", "向前填充 (ffill)", "向後填充 (bfill)", "線性插值", "平均值填充"),
    key='pages_2_impute_method'
)

enable_smoothing = st.sidebar.checkbox("啟用數據平滑", key='pages_2_enable_smoothing')
smoothing_method = None
smoothing_window = 0

if enable_smoothing:
    smoothing_method = st.sidebar.selectbox(
        "平滑方法:",
        ("移動平均", "指數加權平均"),
        key='pages_2_smoothing_method'
    )
    smoothing_window = st.sidebar.number_input(
        "平滑窗口大小:",
        min_value=1, value=7, help="窗口大小會影響平滑程度。例如，每日數據選擇 7 表示取 7 天的平均值。", key='pages_2_smoothing_window'
    )

# --- 布林通道設定 ---
st.sidebar.markdown("---")
st.sidebar.subheader("布林通道設定")
enable_bollinger_bands = st.sidebar.checkbox("顯示布林通道", key='pages_2_enable_bollinger_bands')
bollinger_period = 0
bollinger_std_dev = 0.0
if enable_bollinger_bands:
    bollinger_period = st.sidebar.number_input("布林通道週期:", min_value=2, max_value=60, value=20, step=1, help="布林通道週期需大於 1。", key='pages_2_bollinger_period')
    bollinger_std_dev = st.sidebar.slider("布林通道標準差倍數:", min_value=1.0, max_value=3.0, value=2.0, step=0.1, key='pages_2_bollinger_std_dev')
    if bollinger_period < 2:
        st.sidebar.warning("布林通道週期需大於 1，已自動禁用。")
        enable_bollinger_bands = False

# --- 趨勢/異常事件偵測控制項 ---
st.sidebar.markdown("---")
st.sidebar.subheader("趨勢/異常事件偵測")
enable_cp_detection = st.sidebar.checkbox("啟用趨勢變點偵測", key='pages_2_enable_cp_detection')
cp_penalty = 0
if enable_cp_detection:
    if not ruptures_available:
        st.sidebar.error("請先安裝 ruptures 庫以使用變點偵測。")
        enable_cp_detection = False
    else:
        cp_penalty = st.sidebar.number_input("變點偵測懲罰值 (penalty):", min_value=1, max_value=500, value=10, step=1, help="值越大，偵測到的變點越少。", key='pages_2_cp_penalty')
        if cp_penalty <= 0:
            st.sidebar.warning("懲罰值必須大於0。")
            enable_cp_detection = False

enable_anomaly_detection = st.sidebar.checkbox("啟用異常事件偵測", key='pages_2_enable_anomaly_detection')
anomaly_threshold_std = 0.0
if enable_anomaly_detection:
    anomaly_threshold_std = st.sidebar.slider("異常偵測閾值 (標準差倍數):", min_value=1.0, max_value=5.0, value=2.0, step=0.1, help="數據點與移動平均的標準差倍數，超出此範圍則視為異常。", key='pages_2_anomaly_threshold_std')
    if anomaly_threshold_std <= 0:
        st.sidebar.warning("閾值必須大于0。")
        enable_anomaly_detection = False

# --- 資料載入與分析結果顯示 ---
if 'current_report_data_pages2' not in st.session_state:
    st.session_state.current_report_data_pages2 = None
if 'current_report_params_pages2' not in st.session_state:
    st.session_state.current_report_params_pages2 = (None, None, None, None)

if submitted or (st.session_state.current_report_data_pages2 is not None and \
                    st.session_state.current_report_params_pages2 == (station, year, month, chart_type)):
    
    if submitted or st.session_state.current_report_params_pages2 != (station, year, month, chart_type):
        
        current_station, current_year, current_month, current_chart_type = station, year, month, chart_type
        current_station_name = get_station_name_from_id(current_station)

        with st.spinner(f"正在載入 {current_station_name} 在 {current_year}年 的資料..."):
            df_year = load_year_data(base_data_path, current_station, current_year)
            
        if df_year is None or df_year.empty:
            st.error(f"❌ 找不到 {current_station_name} 在 {current_year}年 的任何資料。")
            st.session_state.current_report_data_pages2 = None
            st.stop()

        time_range_str = f"{current_year}年 全年度" if current_month == 0 else f"{current_year}年{current_month}月"
        df_month_original = df_year if current_month == 0 else df_year[df_year['time'].dt.month == current_month]

        if df_month_original.empty:
            st.error(f"❌ 找不到 {current_station_name} 在 {time_range_str} 的資料。")
            st.session_state.current_report_data_pages2 = None
            st.stop()

        df_processed = df_month_original.copy()

        # 1. 處理缺失值
        if impute_method != "不處理":
            numeric_cols_for_impute = df_processed.select_dtypes(include=np.number).columns
            if impute_method == "向前填充 (ffill)":
                df_processed[numeric_cols_for_impute] = df_processed[numeric_cols_for_impute].ffill()
            elif impute_method == "向後填充 (bfill)":
                df_processed[numeric_cols_for_impute] = df_processed[numeric_cols_for_impute].bfill()
            elif impute_method == "線性插值":
                df_processed[numeric_cols_for_impute] = df_processed[numeric_cols_for_impute].interpolate(method='linear')
            elif impute_method == "平均值填充":
                df_processed[numeric_cols_for_impute] = df_processed[numeric_cols_for_impute].fillna(df_processed[numeric_cols_for_impute].mean())
        
        # 2. 數據平滑 (*** 已依你的要求修改此區塊 ***)
        if enable_smoothing and smoothing_window >= 1:
            numeric_cols_for_smooth = df_processed.select_dtypes(include=np.number).columns
            if not numeric_cols_for_smooth.empty:
                st.info(f"正在使用窗口為 {smoothing_window} 的 '{smoothing_method}' 進行平滑處理...")
                
                if smoothing_method == "移動平均":
                    df_processed[numeric_cols_for_smooth] = df_processed[numeric_cols_for_smooth].rolling(window=smoothing_window, min_periods=1, center=True).mean()
                elif smoothing_method == "指數加權平均":
                    df_processed[numeric_cols_for_smooth] = df_processed[numeric_cols_for_smooth].ewm(span=smoothing_window, adjust=False, min_periods=1).mean()
                
                # 根據你的要求，我們不再移除包含 NaN 的資料列。
                # 請注意：這可能會導致圖表因缺乏有效數據點而顯示為空白。
                # df_processed.dropna(subset=numeric_cols_for_smooth, inplace=True)

            else:
                st.warning("沒有數值型數據可供平滑。")
                
        df_display = df_processed
        change_points_dict = {}
        anomaly_points_dict = {}

        if 'time' not in df_display.columns:
            df_display.reset_index(inplace=True)

        if 'time' not in df_display.columns:
            st.error("❌ 致命錯誤：資料中缺少 'time' 時間欄位。")
            st.stop()

        if enable_cp_detection and ruptures_available:
            for col in df_display.select_dtypes(include=np.number).columns:
                series_data = df_display[col].dropna().values
                if len(series_data) > (cp_penalty or 1):
                    algo = rpt.Pelt(model="rbf", jump=1, min_size=int(cp_penalty/2) or 1).fit(series_data)
                    try:
                        result = algo.predict(pen=cp_penalty)
                        change_points_dict[col] = [df_display['time'].iloc[idx] for idx in result if idx < len(df_display['time'])]
                    except Exception as e:
                        st.warning(f"警告：參數 '{PARAMETER_INFO.get(col, {}).get('display_zh', col)}' 的變點偵測失敗: {e}。")

        if enable_anomaly_detection:
            for col in df_display.select_dtypes(include=np.number).columns:
                series_data = df_display[col].dropna()
                if len(series_data) > 1 and series_data.std() > 1e-9:
                    rolling_mean = series_data.rolling(window=7, min_periods=1).mean()
                    rolling_std = series_data.rolling(window=7, min_periods=1).std().fillna(0)
                    anomalies_idx = series_data[abs(series_data - rolling_mean) > (anomaly_threshold_std * rolling_std)].index
                    anomaly_points_dict[col] = df_display['time'].loc[anomalies_idx].tolist()

        st.session_state.current_report_data_pages2 = {
            'df_display': df_display, 'df_month_original': df_month_original, 'time_range_str': time_range_str,
            'current_station': current_station_name, 'current_year': current_year, 'current_month': current_month,
            'chart_type': current_chart_type, 'change_points_dict': change_points_dict, 'anomaly_points_dict': anomaly_points_dict
        }
        st.session_state.current_report_params_pages2 = (current_station, current_year, current_month, current_chart_type)
        st.success(f"✅ 已成功載入並處理 **{current_station_name}** 在 **{time_range_str}** 的資料！")

    report_data = st.session_state.current_report_data_pages2
    if not report_data or report_data.get('df_display') is None:
         st.info("請點擊 '產生統計報告' 按鈕以開始分析。")
         st.stop()

    df_display = report_data['df_display']
    df_month_original = report_data['df_month_original']
    time_range_str = report_data['time_range_str']
    current_station = report_data['current_station']
    current_year = report_data['current_year']
    current_month = report_data['current_month']
    chart_type = report_data['chart_type']
    change_points_dict = report_data.get('change_points_dict', {})
    anomaly_points_dict = report_data.get('anomaly_points_dict', {})

    if df_display.empty:
        st.warning("數據載入或處理後為空，請重新選擇並生成報告。")
        st.stop()
    
    fig_wave, fig_wind, fig_weather, fig_pie = None, None, None, None

    # 顯示資料品質提示
    st.markdown("---")
    st.subheader("數據品質概覽")
    # 注意：analyze_data_quality 現在可能會處理包含 NaN 的數據
    quality_report = analyze_data_quality(df_display) 

    if quality_report.get('total_records') == 0:
        st.info("本期無數據可供分析。")
    else:
        missing_info = quality_report.get('missing_report')
        outlier_info = quality_report.get('outlier_report')
        has_issues = False
        if missing_info:
            st.warning("⚠️ **部分參數存在缺失數據！**")
            has_issues = True
            for param, data in missing_info.items():
                st.write(f"- **{PARAMETER_INFO.get(param, {}).get('display_zh', param)}**: 缺失 {data['count']} 筆 ({data['percentage']})")

        if outlier_info:
            if not has_issues: st.warning("⚠️ **部分參數可能存在潛在異常值！**")
            has_issues = True
            for param, data in outlier_info.items():
                st.write(f"- **{PARAMETER_INFO.get(param, {}).get('display_zh', param)}**: 檢測到 {data['count']} 個潛在異常值 ({data['percentage']})")

        if not has_issues:
            st.success("✅ **數據品質良好！** 未檢測到顯著缺失或異常數據。")

    # 缺失數據圓餅圖可視化
    st.subheader("📊 數據完整性與潛在異常值")
    st.write("查看選定參數的數據完整性與潛在異常值比例。")
    
    # 圓餅圖應使用原始數據，以反映真實的數據品質
    numeric_cols_for_pie = df_month_original.select_dtypes(include=np.number).columns.tolist()
    
    if numeric_cols_for_pie:
        selected_pie_param = st.selectbox(
            "選擇參數:", options=numeric_cols_for_pie,
            format_func=lambda x: PARAMETER_INFO.get(x, {}).get('display_zh', x),
            key='pages_2_missing_pie_param_select'
        )

        if selected_pie_param:
            total_records = len(df_month_original)
            missing_count = df_month_original[selected_pie_param].isnull().sum()
            
            outlier_count = 0
            non_na_series = df_month_original[selected_pie_param].dropna()
            if not non_na_series.empty:
                Q1 = non_na_series.quantile(0.25)
                Q3 = non_na_series.quantile(0.75)
                IQR = Q3 - Q1
                if IQR > 0:
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outlier_count = non_na_series[(non_na_series < lower_bound) | (non_na_series > upper_bound)].count()
            
            normal_count = total_records - missing_count - outlier_count
            normal_count = max(0, normal_count)

            if total_records > 0:
                pie_data = {'正常數據': normal_count, '潛在異常值': outlier_count, '缺失數據': missing_count}
                df_pie = pd.DataFrame(pie_data.items(), columns=['類別', '筆數'])
                fig_pie = px.pie(df_pie, values='筆數', names='類別', title=f"{PARAMETER_INFO.get(selected_pie_param, {}).get('display_zh', selected_pie_param)} 數據分佈",
                                 hole=0.4, color_discrete_map={'正常數據': 'lightgreen', '潛在異常值': 'salmon', '缺失數據': 'lightgrey'})
                fig_pie.update_traces(textinfo='percent+label', pull=[0, 0.05, 0.05])
                st.plotly_chart(fig_pie, use_container_width=True)
    
    st.markdown("---")
    with st.expander("什麼是潛在異常值 (Potential Outliers)？"):
        st.write("""
        **潛在異常值** 是指在統計上顯著偏離數據集中大多數值的數據點。本應用程式採用 **IQR (Interquartile Range, 四分位距) 方法** 來定義：
        - **判斷標準**：任何小於 $Q1 - 1.5 \\times IQR$ 或大於 $Q3 + 1.5 \\times IQR$ 的數據點。
        - **請注意**：這是一個統計規則，這些點不一定是錯誤數據，可能只是極端但真實的觀測值。
        """)
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["🌊 波浪資料", "🌪️ 風力資料", "🌦️ 氣象資料", "📋 瀏覽數據"])

    with tab1:
        st.subheader(f"波浪資料 - {chart_type}")
        wave_param_cols = ['Wave_Height_Significant', 'Wave_Mean_Period', 'Wave_Peak_Period', 'Wave_Main_Direction']
        wave_cols_available = [col for col in wave_param_cols if col in df_display.columns and not df_display[col].dropna().empty]
        
        if not wave_cols_available:
            st.warning("本期無有效的波浪資料可供顯示。")
        else:
            selected_wave_param_col = None

            if chart_type == '分佈直方圖':
                selected_wave_param_col = st.selectbox("選擇波浪參數:", wave_cols_available, format_func=lambda x: PARAMETER_INFO.get(x, {}).get('display_zh', x), key='pages_2_hist_wave_param')
                if selected_wave_param_col:
                    fig_wave = px.histogram(df_display, x=selected_wave_param_col, nbins=30, labels={"count": "次數"})
            else: # 時間序列圖表
                wave_cols_numeric = [col for col in wave_param_cols if PARAMETER_INFO.get(col, {}).get('type') == 'linear' and col in wave_cols_available]
                if not wave_cols_numeric:
                    st.warning("無數值型波浪資料可繪製此類圖表。")
                else:
                    selected_wave_param_col = st.selectbox("選擇要繪製的波浪參數:", options=wave_cols_numeric, format_func=lambda x: PARAMETER_INFO.get(x, {}).get('display_zh', x), key=f'pages_2_ts_wave_select')
                    
                    if selected_wave_param_col:
                        plot_labels = {'time': "時間", selected_wave_param_col: f"{PARAMETER_INFO.get(selected_wave_param_col, {}).get('display_zh', '')} ({PARAMETER_INFO.get(selected_wave_param_col, {}).get('unit', '')})"}
                        
                        if chart_type == '折線圖':
                            fig_wave = px.line(df_display, x='time', y=selected_wave_param_col, labels=plot_labels, markers=True)
                        elif chart_type == '散佈圖 (含趨勢線)':
                            fig_wave = px.scatter(df_display, x='time', y=selected_wave_param_col, labels=plot_labels, trendline="ols", trendline_color_override="red")
                        elif chart_type == '面積圖':
                            fig_wave = px.area(df_display, x='time', y=selected_wave_param_col, labels=plot_labels)

            if fig_wave:
                if chart_type != '分佈直方圖' and enable_bollinger_bands and selected_wave_param_col:
                    pass
                if chart_type != '分佈直方圖' and enable_cp_detection and selected_wave_param_col in change_points_dict:
                    for cp_time in change_points_dict[selected_wave_param_col]:
                        fig_wave.add_vline(x=cp_time, line_width=1.5, line_dash="dash", line_color="grey", annotation_text="變點")
                if chart_type != '分佈直方圖' and enable_anomaly_detection and selected_wave_param_col in anomaly_points_dict:
                    anomalies_df = df_display[df_display['time'].isin(anomaly_points_dict[selected_wave_param_col])]
                    if not anomalies_df.empty:
                        fig_wave.add_trace(go.Scatter(x=anomalies_df['time'], y=anomalies_df[selected_wave_param_col], mode='markers', marker=dict(symbol='circle', size=10, color='red', line=dict(width=1, color='DarkRed')), name='異常點'))
                st.plotly_chart(fig_wave, use_container_width=True)

    with tab2:
        st.subheader(f"風力資料 - {chart_type}")
        wind_param_cols = ['Wind_Speed', 'Wind_Gust_Speed', 'Wind_Direction']
        wind_cols_available = [col for col in wind_param_cols if col in df_display.columns and not df_display[col].dropna().empty]

        if not wind_cols_available:
            st.warning("本期無有效的風力資料可供顯示。")
        else:
            selected_wind_param_col = None
            if chart_type == '分佈直方圖':
                selected_wind_param_col = st.selectbox("選擇風力參數:", wind_cols_available, format_func=lambda x: PARAMETER_INFO.get(x, {}).get('display_zh', x), key='pages_2_hist_wind_param')
                if selected_wind_param_col:
                    fig_wind = px.histogram(df_display, x=selected_wind_param_col, nbins=30, labels={"count": "次數"})
            else:
                wind_cols_numeric = [col for col in wind_param_cols if PARAMETER_INFO.get(col, {}).get('type') == 'linear' and col in wind_cols_available]
                if not wind_cols_numeric:
                    st.warning("無數值型風力資料可繪製此類圖表。")
                else:
                    selected_wind_param_col = st.selectbox("選擇要繪製的風力參數:", options=wind_cols_numeric, format_func=lambda x: PARAMETER_INFO.get(x, {}).get('display_zh', x), key=f'pages_2_ts_wind_select')
                    
                    if selected_wind_param_col:
                        plot_labels = {'time': "時間", selected_wind_param_col: f"{PARAMETER_INFO.get(selected_wind_param_col, {}).get('display_zh', '')} ({PARAMETER_INFO.get(selected_wind_param_col, {}).get('unit', '')})"}
                        if chart_type == '折線圖':
                            fig_wind = px.line(df_display, x='time', y=selected_wind_param_col, labels=plot_labels, markers=True)
                        elif chart_type == '散佈圖 (含趨勢線)':
                            fig_wind = px.scatter(df_display, x='time', y=selected_wind_param_col, labels=plot_labels, trendline="ols", trendline_color_override="green")
                        elif chart_type == '面積圖':
                            fig_wind = px.area(df_display, x='time', y=selected_wind_param_col, labels=plot_labels)
            if fig_wind:
                st.plotly_chart(fig_wind, use_container_width=True)

    with tab3:
        st.subheader(f"氣象資料 - {chart_type}")
        weather_param_cols = ['Air_Temperature', 'Sea_Temperature', 'Air_Pressure']
        weather_cols_available = [col for col in weather_param_cols if col in df_display.columns and not df_display[col].dropna().empty]

        if not weather_cols_available:
            st.warning("本期無有效的氣象資料可供顯示。")
        else:
            selected_weather_param_col = None
            if chart_type == '分佈直方圖':
                selected_weather_param_col = st.selectbox("選擇氣象參數:", weather_cols_available, format_func=lambda x: PARAMETER_INFO.get(x, {}).get('display_zh', x), key='pages_2_hist_weather_param')
                if selected_weather_param_col:
                    fig_weather = px.histogram(df_display, x=selected_weather_param_col, nbins=30, labels={"count": "次數"})
            else:
                weather_cols_numeric = [col for col in weather_param_cols if PARAMETER_INFO.get(col, {}).get('type') == 'linear' and col in weather_cols_available]
                if not weather_cols_numeric:
                    st.warning("無數值型氣象資料可繪製此類圖表。")
                else:
                    selected_weather_param_col = st.selectbox("選擇要繪製的氣象參數:", options=weather_cols_numeric, format_func=lambda x: PARAMETER_INFO.get(x, {}).get('display_zh', x), key=f'pages_2_ts_weather_select')
                    
                    if selected_weather_param_col:
                        plot_labels = {'time': "時間", selected_weather_param_col: f"{PARAMETER_INFO.get(selected_weather_param_col, {}).get('display_zh', '')} ({PARAMETER_INFO.get(selected_weather_param_col, {}).get('unit', '')})"}
                        if chart_type == '折線圖':
                            fig_weather = px.line(df_display, x='time', y=selected_weather_param_col, labels=plot_labels, markers=True)
                        elif chart_type == '散佈圖 (含趨勢線)':
                            fig_weather = px.scatter(df_display, x='time', y=selected_weather_param_col, labels=plot_labels, trendline="ols", trendline_color_override="purple")
                        elif chart_type == '面積圖':
                            fig_weather = px.area(df_display, x='time', y=selected_weather_param_col, labels=plot_labels)
            if fig_weather:
                st.plotly_chart(fig_weather, use_container_width=True)

    with tab4:
        st.subheader("瀏覽數據")
        st.write("顯示的數據為經過側邊欄所有預處理選項（缺失值填充、平滑等）後的最終結果。")
        st.dataframe(df_display)
        st.download_button("📥 下載顯示數據 (CSV)", convert_df_to_csv(df_display), f"data_{station}_{year}{month:02d}.csv", "text/csv")

    # ====================================================================
    #  報告下載區 (包含個別下載與打包下載)
    # ====================================================================
    st.markdown("---")
    st.header("📦 報告下載區")

    # 生成摘要報告 (TXT)
    summary_str = f"""單站資料探索報告
================================
測站ID: {current_station}
時間範圍: {time_range_str}
報告生成時間: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
================================

數據預處理設定
--------------------------------
缺失值處理: {impute_method}
數據平滑: {'啟用 (方法: ' + smoothing_method + ', 窗口: ' + str(smoothing_window) + ')' if enable_smoothing else '停用'}
================================

數據品質概覽
--------------------------------
"""
    if quality_report.get('total_records', 0) > 0:
        summary_str += f"總筆數: {quality_report.get('total_records')}\n"
        if quality_report.get('missing_report'):
            summary_str += "\n缺失值報告:\n"
            for param, data in quality_report['missing_report'].items():
                param_name = PARAMETER_INFO.get(param, {}).get('display_zh', param)
                summary_str += f"- {param_name}: 缺失 {data['count']} 筆 ({data['percentage']})\n"
        if quality_report.get('outlier_report'):
            summary_str += "\n潛在異常值報告 (IQR方法):\n"
            for param, data in quality_report['outlier_report'].items():
                param_name = PARAMETER_INFO.get(param, {}).get('display_zh', param)
                summary_str += f"- {param_name}: 檢測到 {data['count']} 個 ({data['percentage']})\n"
    else:
        summary_str += "本期無數據可供分析。\n"
    
    summary_bytes = summary_str.encode('utf-8')

    with st.expander("📂 **個別檔案下載 (點此展開)**"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("##### **圖表 (HTML)**")
            if fig_wave: st.download_button("📥 下載波浪資料圖", fig_wave.to_html().encode('utf-8'), f"chart_wave_{current_station}_{time_range_str.replace(' ', '')}.html", "text/html")
            if fig_wind: st.download_button("📥 下載風力資料圖", fig_wind.to_html().encode('utf-8'), f"chart_wind_{current_station}_{time_range_str.replace(' ', '')}.html", "text/html")
            if fig_weather: st.download_button("📥 下載氣象資料圖", fig_weather.to_html().encode('utf-8'), f"chart_weather_{current_station}_{time_range_str.replace(' ', '')}.html", "text/html")
            if fig_pie: st.download_button("📥 下載數據完整性圖", fig_pie.to_html().encode('utf-8'), f"chart_quality_{current_station}_{time_range_str.replace(' ', '')}.html", "text/html")
        with col2:
            st.markdown("##### **數據 (CSV)**")
            st.download_button("📥 下載原始數據", convert_df_to_csv(df_month_original), f"data_original_{current_station}_{time_range_str.replace(' ', '')}.csv", "text/csv")
            st.download_button("📥 下載處理後數據", convert_df_to_csv(df_display), f"data_processed_{current_station}_{time_range_str.replace(' ', '')}.csv", "text/csv")
        with col3:
            st.markdown("##### **摘要 (TXT)**")
            st.download_button("📥 下載文字摘要報告", summary_bytes, f"summary_{current_station}_{time_range_str.replace(' ', '')}.txt", "text/plain")
    
    st.markdown("---")
    st.subheader("🗂️ 一鍵打包下載 (ZIP)")
    st.write("點擊下方按鈕，即可將上述所有圖表、數據和摘要打包成一個 ZIP 檔案下載。")

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
        if fig_wave: zip_file.writestr(f"charts/wave_chart.html", fig_wave.to_html())
        if fig_wind: zip_file.writestr(f"charts/wind_chart.html", fig_wind.to_html())
        if fig_weather: zip_file.writestr(f"charts/weather_chart.html", fig_weather.to_html())
        if fig_pie: zip_file.writestr(f"charts/quality_chart.html", fig_pie.to_html())
        zip_file.writestr(f"data/processed_data.csv", convert_df_to_csv(df_display))
        zip_file.writestr(f"data/original_data.csv", convert_df_to_csv(df_month_original))
        zip_file.writestr("summary_report.txt", summary_bytes)

    st.download_button(label="📥 **點此下載打包好的 ZIP 檔案**", data=zip_buffer.getvalue(), file_name=f"report_{current_station}_{time_range_str.replace(' ', '')}.zip", mime="application/zip")
