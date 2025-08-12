import streamlit as st
import pandas as pd
import plotly.express as px
from utils.helpers import get_station_name_from_id, load_year_data, PARAMETER_INFO, initialize_session_state
import io
import zipfile

# 設定為寬螢幕模式
st.set_page_config(layout="wide")

# --- 狀態初始化 ---
if 'analysis_run' not in st.session_state:
    st.session_state.analysis_run = False
    st.session_state.results = {}

# --- 頁面標題 ---
initialize_session_state()
st.title("📈 多站時間序列比較")
st.write("同時檢視多個測站在特定年份，同一參數的時間序列數據，以便進行趨勢比較。")
st.markdown("---")

locations = st.session_state.get('locations', [])
base_data_path = st.session_state.get('base_data_path', '')
available_years = st.session_state.get('available_years', [])

if not locations:
    st.warning("請返回主頁面以載入測站列表。")
    st.stop()
if not available_years:
    st.warning("沒有偵測到任何可用的年份資料，請檢查資料夾設定或返回主頁面重新載入。")
    st.stop()

# --- 側邊欄控制項 (已移除所有會重設狀態的 on_change) ---
with st.sidebar:
    st.header("比較設定")

    def toggle_all_stations():
        if st.session_state.get('pages_9_select_all_checkbox', False):
            st.session_state['pages_9_multi_station_select'] = locations
        else:
            st.session_state['pages_9_multi_station_select'] = []

    if 'pages_9_multi_station_select' not in st.session_state:
        st.session_state['pages_9_multi_station_select'] = [locations[0]] if locations else []

    st.checkbox("全選/取消全選所有測站", key='pages_9_select_all_checkbox', on_change=toggle_all_stations,
                        value=(len(st.session_state.get('pages_9_multi_station_select', [])) == len(locations) and bool(locations)))
    
    selected_stations = st.multiselect("選擇要比較的測站:", options=locations, key='pages_9_multi_station_select', format_func=get_station_name_from_id)
    
    default_year_index = len(available_years) - 1 if available_years else 0
    selected_year = st.selectbox("選擇年份:", options=available_years, index=default_year_index, key='pages_9_multi_year_select')

    comparable_params = {f"{info['display_zh']} ({info['unit']})": col_name for col_name, info in PARAMETER_INFO.items() if info.get('type') == 'linear'}
    sorted_param_options = sorted(list(comparable_params.keys()))
    selected_param_display = st.selectbox("選擇要比較的參數:", options=sorted_param_options, key='pages_9_multi_param_select')
    selected_param_col = comparable_params[selected_param_display]

    run_button_clicked = st.button("📊 執行比較", key='pages_9_multi_compare_button', use_container_width=True)

# --- 核心邏輯：點擊按鈕時計算，並在同一次刷新中顯示 ---
if run_button_clicked:
    if not selected_stations:
        st.warning("請至少選擇一個測站進行比較。")
    else:
        with st.spinner("正在執行分析，請稍候..."):
            all_stations_data = []
            for station_id in selected_stations:
                station_name = get_station_name_from_id(station_id)
                df_station = load_year_data(base_data_path, station_id, selected_year)
                if df_station is not None and not df_station.empty and selected_param_col in df_station.columns:
                    df_filtered = df_station[['time', selected_param_col]].dropna()
                    if not df_filtered.empty:
                        df_filtered['測站'] = station_name
                        all_stations_data.append(df_filtered)
            
            if not all_stations_data:
                st.error("沒有找到任何可供比較的有效數據。請檢查您的選擇或資料是否存在。")
                st.session_state.analysis_run = False
                st.session_state.results = {}
            else:
                combined_df = pd.concat(all_stations_data, ignore_index=True).sort_values(by='time').reset_index(drop=True)
                st.session_state.results = {
                    'combined_df': combined_df,
                    'selected_year': selected_year,
                    'selected_param_display': selected_param_display,
                    'selected_param_col': selected_param_col,
                    'selected_stations': selected_stations
                }
                st.session_state.analysis_run = True

# --- 顯示結果區塊 ---
if st.session_state.get('analysis_run', False):
    # 檢查 session_state 中是否已有結果
    if 'combined_df' not in st.session_state.get('results', {}):
        st.warning("請點擊「執行比較」以載入數據。")
    else:
        results = st.session_state.results
        
        # 檢查當前選擇是否與已顯示結果的參數相符
        is_stale = (
            set(results.get('selected_stations', [])) != set(selected_stations) or
            results.get('selected_year') != selected_year or
            results.get('selected_param_display') != selected_param_display
        )
        if is_stale:
            st.warning("⚠️ 您已更改側邊欄的設定，目前的分析結果可能已過時。請重新點擊「執行比較」以更新圖表。")

        # 從 session_state 讀取結果以顯示
        combined_df = results['combined_df']
        result_year = results['selected_year']
        result_param_display = results['selected_param_display']
        result_param_col = results['selected_param_col']
        result_stations = [get_station_name_from_id(station) for station in results['selected_stations']]
        
        st.subheader(f"分析結果：{result_year}年 - {result_param_display}")
        
        main_tabs = st.tabs(["📈 **數據儀表板**", "📊 **趨勢圖表**", "📄 **詳細數據**", "📝 **數據摘要**"])

        # TAB 1: 數據儀表板
        with main_tabs[0]:
            time_min, time_max = combined_df['time'].min(), combined_df['time'].max()
            all_diffs = combined_df.sort_values('time')['time'].diff()
            positive_diffs = all_diffs[all_diffs.dt.total_seconds() > 0]
            time_diff = positive_diffs.median() if not positive_diffs.empty else None
            quality_data = []
            for station in result_stations:
                station_df = combined_df[combined_df['測站'] == station]
                actual_points = len(station_df)
                stats = station_df[result_param_col].describe()
                num_outliers = 0
                if actual_points > 1 and pd.api.types.is_numeric_dtype(station_df[result_param_col]):
                    Q1, Q3 = stats.get('25%'), stats.get('75%')
                    if Q1 is not None and Q3 is not None:
                        IQR = Q3 - Q1
                        if IQR > 0:
                            lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
                            num_outliers = station_df[(station_df[result_param_col] < lower_bound) | (station_df[result_param_col] > upper_bound)].shape[0]
                completeness = 100.0
                expected_points = actual_points
                if time_diff:
                    expected_points = len(pd.date_range(start=time_min, end=time_max, freq=time_diff))
                    completeness = (actual_points / expected_points) * 100 if expected_points > 0 else 0
                quality_data.append({"測站": station, "總有效點數": actual_points, "正常數據點": actual_points - num_outliers, "異常值點數": num_outliers, "異常值比例 (%)": (num_outliers / actual_points) * 100 if actual_points > 0 else 0, "數據完整度 (%)": completeness, "最小值": stats.get('min'), "平均值": stats.get('mean'), "最大值": stats.get('max'),})
            quality_df = pd.DataFrame(quality_data)

            with st.container(border=True):
                st.subheader("1. 時間與完整度分析")
                st.markdown("---")
                total_points = quality_df['總有效點數'].sum()
                total_expected = len(pd.date_range(start=time_min, end=time_max, freq=time_diff)) if time_diff else total_points
                metric_cols = st.columns(3)
                metric_cols[0].metric("數據時間範圍", f"{time_min.strftime('%y/%m/%d')} - {time_max.strftime('%y/%m/%d')}")
                if time_diff:
                    metric_cols[1].metric("推斷資料頻率", f"~ {pd.to_timedelta(time_diff).total_seconds() / 60:.1f} 分鐘/筆")
                    metric_cols[2].metric("理論總點數", f"{total_expected:,}")
                else:
                    metric_cols[1].metric("推斷資料頻率", "無法推斷")
                total_normal, total_outliers = quality_df['正常數據點'].sum(), quality_df['異常值點數'].sum()
                total_missing = max(0, total_expected - total_points)
                pie_data = pd.DataFrame({'類型': ['正常數據', '異常值', '理論缺失'], '數量': [total_normal, total_outliers, total_missing]})
                pie_data = pie_data[pie_data['數量'] > 0]
                if not pie_data.empty:
                    st.plotly_chart(px.pie(pie_data, values='數量', names='類型', title='整體數據結構分佈', hole=0.4, color_discrete_map={'正常數據': '#28a745', '異常值': '#ffc107', '理論缺失': '#dc3545'}), use_container_width=True)

            with st.container(border=True):
                st.subheader("2. 測站數據品質探查")
                explore_tabs = st.tabs(["數據統計表", "異常值視覺化 (箱形圖)"])
                with explore_tabs[0]:
                    st.dataframe(quality_df, use_container_width=True, hide_index=True, column_config={"異常值比例 (%)": st.column_config.ProgressColumn(format="%.2f%%"), "數據完整度 (%)": st.column_config.ProgressColumn(format="%.2f%%"), "平均值": st.column_config.NumberColumn(format="%.2f"), "最小值": st.column_config.NumberColumn(format="%.2f"), "最大值": st.column_config.NumberColumn(format="%.2f"),})
                with explore_tabs[1]:
                    focus_station = st.selectbox("選擇要聚焦的測站：", options=['顯示所有測站'] + result_stations)
                    df_to_plot = combined_df if focus_station == '顯示所有測站' else combined_df[combined_df['測站'] == focus_station]
                    title = f"{result_param_display} 數據分佈 ({focus_station})"
                    st.plotly_chart(px.box(df_to_plot, x='測站', y=result_param_col, color='測站', title=title), use_container_width=True)

        # TAB 2: 趨勢圖表
        with main_tabs[1]:
            st.subheader("多樣化趨勢視覺化")
            chart_type = st.radio("選擇圖表類型：", options=["線形圖", "面積圖", "散佈圖", "熱力圖"], horizontal=True, key="chart_type_selector")
            y_axis_title = f"{PARAMETER_INFO[result_param_col]['display_zh']} ({PARAMETER_INFO[result_param_col]['unit']})"
            fig = None
            if chart_type == "線形圖":
                fig = px.line(combined_df, x='time', y=result_param_col, color='測站', title=f"{result_year} 年 {result_param_display} 趨勢 (線形圖)", labels={'time': '時間', result_param_col: y_axis_title, '測站': '測站'})
            elif chart_type == "面積圖":
                fig = px.area(combined_df, x='time', y=result_param_col, color='測站', title=f"{result_year} 年 {result_param_display} 趨勢 (面積圖)", labels={'time': '時間', result_param_col: y_axis_title, '測站': '測站'})
            elif chart_type == "散佈圖":
                fig = px.scatter(combined_df, x='time', y=result_param_col, color='測站', title=f"{result_year} 年 {result_param_display} 數據分佈 (散佈圖)", labels={'time': '時間', result_param_col: y_axis_title, '測站': '測站'}, opacity=0.6)
            elif chart_type == "熱力圖":
                freq_opts = {'D': '每日平均', 'W': '每週平均', 'M': '每月平均'}
                freq = st.selectbox("選擇時間聚合頻率：", options=list(freq_opts.keys()), format_func=lambda x: freq_opts[x])
                try:
                    resampled = combined_df.set_index('time').groupby('測站')[result_param_col].resample(freq).mean().reset_index()
                    pivoted = resampled.pivot(index='測站', columns='time', values=result_param_col)
                    fig = px.imshow(pivoted, labels=dict(x="時間", y="測站", color=y_axis_title), aspect="auto", title=f"{result_year} 年 {result_param_display} 熱力圖 ({freq_opts[freq]})")
                except Exception as e: st.error(f"繪製熱力圖時發生錯誤: {e}")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                st.session_state.results['fig'] = fig

        # TAB 3 & 4
        with main_tabs[2]:
            st.dataframe(combined_df, use_container_width=True)
        with main_tabs[3]:
            summary_df = combined_df.groupby('測站')[result_param_col].describe()
            st.dataframe(summary_df.rename(columns={'count': '點數', 'mean': '平均值', 'std': '標準差', 'min': '最小值', 'max': '最大值'}), use_container_width=True)
            st.session_state.results['summary_df'] = summary_df

        # 下載區塊
        with st.container(border=True):
            st.subheader("📥 下載報告與數據")
            base_filename = f"multi_station_{result_param_col}_{result_year}"
            csv_data = combined_df.to_csv(index=False).encode('utf-8-sig')
            summary_df_exists = 'summary_df' in results
            txt_data = results['summary_df'].to_string().encode('utf-8') if summary_df_exists else b""
            html_data = results.get('fig').to_html().encode('utf-8') if 'fig' in results else b""
            dl_cols = st.columns(3)
            dl_cols[0].download_button("下載數據 (CSV)", csv_data, f"{base_filename}_data.csv", "text/csv", use_container_width=True)
            dl_cols[1].download_button("下載摘要 (TXT)", txt_data, f"{base_filename}_summary.txt", "text/plain", use_container_width=True, disabled=not summary_df_exists)
            dl_cols[2].download_button("下載圖表 (HTML)", html_data, f"{base_filename}_chart.html", "text/html", use_container_width=True, disabled='fig' not in results)
            st.divider()
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                zf.writestr(f"{base_filename}_data.csv", csv_data)
                if summary_df_exists: zf.writestr(f"{base_filename}_summary.txt", txt_data)
                if 'fig' in results: zf.writestr(f"{base_filename}_chart.html", html_data)
            st.download_button("🚀 一鍵打包下載 (ZIP)", zip_buffer.getvalue(), f"{base_filename}_package.zip", "application/zip", use_container_width=True)
else:
    # 歡迎畫面
    with st.container(border=True):
        st.info("👋 **歡迎使用多站比較工具！**")
        st.write("請在左方側邊欄選擇您想比較的 **測站**、**年份** 和 **參數**，然後點擊「**執行比較**」按鈕，即可開始分析。")
