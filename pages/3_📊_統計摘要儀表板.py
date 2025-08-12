import streamlit as st
import pandas as pd
import plotly.express as px
from utils.helpers import get_station_name_from_id, initialize_session_state, load_year_data, convert_df_to_csv, PARAMETER_INFO, analyze_data_quality
import io
import zipfile

initialize_session_state()
st.title("📊 統計摘要儀表板")
st.write("快速生成指定測站與時間範圍內的數據統計摘要報告。")
st.markdown("---")

# 從 session_state 讀取共享資料
locations = st.session_state.get('locations', [])
base_data_path = st.session_state.get('base_data_path', '')
all_available_years = st.session_state.get('available_years', [])

if not locations:
    st.warning("請返回主頁面以載入測站列表。")
    st.stop()

if not all_available_years:
    st.warning("沒有偵測到任何可用的年份資料，請檢查資料夾設定或返回主頁面重新載入。")
    st.stop()

# --- 選擇器區塊 ---
col1, col2 = st.columns(2)
with col1:
    station_selected = st.selectbox("選擇測站", locations, key='pages_3_db_station_form', format_func=get_station_name_from_id)
    station_selected_name = get_station_name_from_id(station_selected)

@st.cache_data
def get_station_specific_years(station, years_to_check, data_path):
    valid_years = []
    for year in years_to_check:
        df_check = load_year_data(data_path, station, year)
        if df_check is not None and not df_check.empty:
            valid_years.append(year)
    return sorted(valid_years, reverse=True)

with st.spinner(f"正在查詢 {station_selected_name} 的可用年份..."):
    station_years = get_station_specific_years(station_selected, all_available_years, base_data_path)

if not station_years:
    with col2:
        st.selectbox("選擇年份", ["該測站無資料"], disabled=True)
    st.error(f"❌ 找不到測站 **{station_selected_name}** 的任何年份資料。")
    st.info("請嘗試選擇其他測站。")
    st.stop()

with col2:
    year_selected = st.selectbox("選擇年份", station_years, index=0, key='pages_3_db_year_form')

with st.spinner(f"正在載入 {station_selected} 在 {year_selected}年 的資料..."):
    df_year = load_year_data(base_data_path, station_selected, year_selected)

if df_year is None or df_year.empty or 'time' not in df_year.columns:
    st.error(f"❌ 載入資料時發生預期外的錯誤。找不到 {station_selected_name} 在 {year_selected}年 的有效資料或時間欄位。")
    st.session_state.current_report_data = None
    st.stop()
    
valid_months = sorted(df_year['time'].dt.month.unique())
month_options = {0: "全年"}
month_options.update({m: f"{m}月" for m in valid_months})

with st.form("main_dashboard_form"):
    month_selected = st.selectbox("選擇月份", list(month_options.keys()), format_func=lambda x: month_options[x], key='pages_3_db_month_form')
    submitted = st.form_submit_button("🚀 產生統計報告")

if 'current_report_data' not in st.session_state:
    st.session_state.current_report_data = None
if 'current_report_params' not in st.session_state:
    st.session_state.current_report_params = (None, None, None)

if submitted or st.session_state.current_report_params != (station_selected, year_selected, month_selected):
    st.session_state.current_report_params = (station_selected, year_selected, month_selected)
    
    with st.spinner("正在產生報告..."):
        df_selection_temp = df_year if month_selected == 0 else df_year[df_year['time'].dt.month == month_selected]
        time_range_str_temp = f"{year_selected}年 全年度" if month_selected == 0 else f"{year_selected}年{month_selected}月"

        if df_selection_temp.empty:
            st.warning(f"🔍 在 {time_range_str_temp} 沒有找到資料。")
            st.session_state.current_report_data = None
        else:
            st.session_state.current_report_data = {
                'df_selection': df_selection_temp,
                'time_range_str': time_range_str_temp
            }
            st.success(f"✅ 已成功載入 **{station_selected_name}** 在 **{time_range_str_temp}** 的資料！")

if st.session_state.current_report_data is not None:
    df_selection = st.session_state.current_report_data['df_selection']
    time_range_str = st.session_state.current_report_data['time_range_str']
    current_station, current_year, current_month = st.session_state.current_report_params
    current_station_name = get_station_name_from_id(current_station)

    if df_selection.empty:
        st.warning("數據載入失敗，請重新選擇並生成報告。")
        st.session_state.current_report_data = None
        st.stop()

    fig_hist, fig_ts, fig_box, fig_pie_combined = None, None, None, None

    st.markdown("---")
    st.subheader("數據品質概覽")
    quality_report = analyze_data_quality(df_selection)
    first_param_key = next(iter(quality_report), None)

    if not first_param_key or quality_report[first_param_key].get('total_records', 0) == 0:
        st.info("本期無數據可供分析。")
    else:
        missing_items = {p: m for p, m in quality_report.items() if m.get('missing_count', 0) > 0}
        outlier_items = {p: m for p, m in quality_report.items() if m.get('outlier_iqr_count', 0) > 0}
        has_issues = False

        if missing_items:
            st.warning("⚠️ **部分參數存在缺失數據！**")
            has_issues = True
            for param, data in missing_items.items():
                st.write(f"- **{PARAMETER_INFO.get(param, {}).get('display_zh', param)}**: 缺失 {data['missing_count']} 筆 ({data['missing_percentage']:.2f}%)")
        
        if outlier_items:
            st.warning(f"⚠️ **部分參數可能存在潛在異常值！** (使用 IQR 方法檢測)")
            has_issues = True
            for param, data in outlier_items.items():
                percentage = (data['outlier_iqr_count'] / data['valid_count']) * 100 if data['valid_count'] > 0 else 0
                st.write(f"- **{PARAMETER_INFO.get(param, {}).get('display_zh', param)}**: 檢測到 {data['outlier_iqr_count']} 個潛在異常值 ({percentage:.2f}%)")

        if not has_issues:
            st.success("✅ **數據品質良好！** 未檢測到顯著缺失或異常數據。")

        st.markdown("##### 視覺化分析")
        params_with_quality_metrics = [p for p, m in quality_report.items() if m.get('is_numeric', False)]
        
        if params_with_quality_metrics:
            selected_param_for_pie = st.selectbox(
                "選擇一個參數來查看其數據品質圓餅圖：",
                options=params_with_quality_metrics,
                format_func=lambda x: PARAMETER_INFO.get(x, {}).get('display_zh', x)
            )

            if selected_param_for_pie:
                param_metrics = quality_report[selected_param_for_pie]
                param_zh = PARAMETER_INFO.get(selected_param_for_pie, {}).get('display_zh', selected_param_for_pie)
                missing_count = param_metrics.get('missing_count', 0)
                outlier_count = param_metrics.get('outlier_iqr_count', 0)
                valid_count = param_metrics.get('valid_count', 0)
                normal_count = valid_count - outlier_count

                if missing_count > 0 or valid_count > 0:
                    pie_data_combined = pd.DataFrame({
                        '類別': ['正常範圍數據', '潛在異常值', '缺失數據'],
                        '數量': [normal_count, outlier_count, missing_count]
                    })
                    pie_data_combined = pie_data_combined[pie_data_combined['數量'] > 0]
                    fig_pie_combined = px.pie(
                        pie_data_combined, names='類別', values='數量',
                        title=f"<b>{param_zh}：整體數據品質分析</b>", hole=0.4,
                        color_discrete_map={'正常範圍數據': '#1E88E5', '潛在異常值': '#D81B60', '缺失數據': '#FFC107'}
                    )
                    pull_values = [0.1 if cat in ['潛在異常值', '缺失數據'] else 0 for cat in pie_data_combined['類別']]
                    fig_pie_combined.update_traces(textinfo='percent+label', pull=pull_values)
                    fig_pie_combined.update_layout(legend_title_text='數據類別', margin=dict(l=10, r=10, t=50, b=10))
                    st.plotly_chart(fig_pie_combined, use_container_width=True)
                else:
                    st.info("此參數無數據可供分析。")

    st.markdown("---")
    st.subheader("📈 關鍵趨勢概覽")
    all_linear_params = [col for col, info in PARAMETER_INFO.items() if info.get('type') == 'linear' and col in df_selection.columns and df_selection[col].dropna().any()]

    if all_linear_params:
        default_trend_params_key = f'pages_3_trend_params_select_default_{current_station}_{current_year}_{current_month}'
        selected_trend_params_english = st.multiselect(
            "選擇要顯示趨勢圖的參數 (最多 3 個)",
            options=all_linear_params,
            default=st.session_state.get(default_trend_params_key, all_linear_params[:min(len(all_linear_params), 2)]),
            format_func=lambda x: PARAMETER_INFO.get(x, {}).get('display_zh', x),
            key=f'pages_3_trend_params_select_multi_{current_station}_{current_year}_{current_month}'
        )
        st.session_state[default_trend_params_key] = selected_trend_params_english

        if selected_trend_params_english:
            display_params_limited = selected_trend_params_english[:3]
            cols_for_trend_charts = st.columns(len(display_params_limited))
            for i, param_col in enumerate(display_params_limited):
                param_zh = PARAMETER_INFO.get(param_col, {}).get('display_zh', param_col)
                param_unit = PARAMETER_INFO.get(param_col, {}).get('unit', '')
                with cols_for_trend_charts[i]:
                    fig_trend = px.line(
                        df_selection, x='time', y=param_col,
                        title=f"{param_zh} 趨勢",
                        labels={'time': '時間', param_col: f"{param_zh} ({param_unit})"}, height=200
                    )
                    fig_trend.update_layout(showlegend=False, margin=dict(l=20, r=20, t=30, b=20))
                    
                    # <<< 修改開始：顯示X軸的時間刻度標籤 >>>
                    # 將 showticklabels 改為 True 來顯示時間
                    # 繼續隱藏軸標題 (title_text="") 以節省空間
                    fig_trend.update_xaxes(showticklabels=True, title_text="")
                    # <<< 修改結束 >>>

                    st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("請選擇至少一個參數以顯示趨勢圖。")
    else:
        st.info("沒有可用的數值型參數來顯示趨勢圖。")

    st.markdown("---")
    numeric_cols = [c for c in df_selection.columns if pd.api.types.is_numeric_dtype(df_selection[c])]
    df_numeric = df_selection[numeric_cols].copy()

    if not df_numeric.empty:
        st.subheader("詳細統計數據")
        stats_df = df_numeric.describe(percentiles=[.25, .5, .75, .9, .95]).T
        stats_df.index = [PARAMETER_INFO.get(idx, {}).get('display_zh', idx) for idx in stats_df.index]
        st.dataframe(stats_df.style.format("{:.2f}"))

        st.subheader("數據分佈視覺化 (箱形圖)")
        df_long = pd.melt(df_numeric, var_name='參數', value_name='數值')
        df_long['參數'] = df_long['參數'].apply(
            lambda x: f"{PARAMETER_INFO.get(x, {}).get('display_zh', x)}{(' (' + PARAMETER_INFO.get(x, {}).get('unit', '') + ')') if PARAMETER_INFO.get(x, {}).get('unit') else ''}")
        fig_box = px.box(df_long, x='參數', y='數值', points='outliers',
                            labels={"參數": "參數", "數值": "數值"},
                            title=f"{current_station_name} 在 {time_range_str} 的數據分佈箱形圖")
        st.plotly_chart(fig_box, use_container_width=True)

        st.subheader("數據趨勢視覺化 (時間序列圖)")
        with st.form("time_series_chart_form"):
            time_series_cols = [col for col in numeric_cols if col != 'time']
            if time_series_cols:
                selected_ts_param_english = st.selectbox(
                    "選擇要顯示時間序列圖的參數", time_series_cols,
                    format_func=lambda x: PARAMETER_INFO.get(x, {}).get('display_zh', x),
                    key=f'ts_param_select_main_{current_station}_{current_year}_{current_month}_final'
                )
                ts_chart_submitted = st.form_submit_button("更新時間序列圖")
                
                if ts_chart_submitted:
                    fig_ts = px.line(df_selection, x='time', y=selected_ts_param_english,
                        title=f"{current_station_name} 在 {time_range_str} 的 {PARAMETER_INFO.get(selected_ts_param_english, {}).get('display_zh', selected_ts_param_english)} 趨勢",
                        labels={"time": "時間", selected_ts_param_english: f"{PARAMETER_INFO.get(selected_ts_param_english, {}).get('display_zh', selected_ts_param_english)} ({PARAMETER_INFO.get(selected_ts_param_english, {}).get('unit', '')})"})
                    fig_ts.update_xaxes(rangeselector=dict(buttons=list([
                            dict(count=1, label="1m", step="month", stepmode="backward"), dict(count=6, label="6m", step="month", stepmode="backward"),
                            dict(count=1, label="YTD", step="year", stepmode="todate"), dict(count=1, label="1y", step="year", stepmode="backward"),
                            dict(step="all")])),
                        rangeslider=dict(visible=True), type="date")
                    st.session_state[f'last_ts_chart_{current_station}_{current_year}_{current_month}'] = fig_ts
                
                if f'last_ts_chart_{current_station}_{current_year}_{current_month}' in st.session_state:
                    st.plotly_chart(st.session_state[f'last_ts_chart_{current_station}_{current_year}_{current_month}'], use_container_width=True)
                else:
                    st.info("請選擇一個參數並點擊「更新時間序列圖」按鈕。")
            else:
                st.info("沒有可繪製時間序列圖的數值型參數。")
        
        st.markdown("---")
        st.subheader("📦 下載分析產出")
        st.write("您可以下載包含所有產出的 .zip 壓縮檔，或分別下載各類型的檔案。")

        raw_data_csv = convert_df_to_csv(df_selection)
        
        stats_df_downloadable = stats_df.copy()
        stats_df_downloadable.index.name = "參數"
        stats_csv = convert_df_to_csv(stats_df_downloadable)

        fig_pie_html = fig_pie_combined.to_html() if fig_pie_combined else ""
        fig_box_html = fig_box.to_html() if fig_box else ""
        fig_ts_html = ""
        if f'last_ts_chart_{current_station}_{current_year}_{current_month}' in st.session_state:
            fig_ts_html = st.session_state[f'last_ts_chart_{current_station}_{current_year}_{current_month}'].to_html()
        
        summary_text_io = io.StringIO()
        summary_text_io.write("====================================================\n")
        summary_text_io.write(f" 統計摘要報告\n")
        summary_text_io.write("====================================================\n\n")
        summary_text_io.write(f"測站: {current_station_name}\n")
        summary_text_io.write(f"時間範圍: {time_range_str}\n\n")
        summary_text_io.write("----------------------------------------------------\n")
        summary_text_io.write(" 1. 數據品質概覽\n")
        summary_text_io.write("----------------------------------------------------\n\n")
        summary_has_issues = False
        summary_missing_items = {p: m for p, m in quality_report.items() if m.get('missing_count', 0) > 0}
        summary_outlier_items = {p: m for p, m in quality_report.items() if m.get('outlier_iqr_count', 0) > 0}
        if summary_missing_items:
            summary_has_issues = True
            summary_text_io.write("⚠️ 部分參數存在缺失數據！\n")
            for param, data in summary_missing_items.items():
                param_zh_txt = PARAMETER_INFO.get(param, {}).get('display_zh', param)
                summary_text_io.write(f"- {param_zh_txt}: 缺失 {data['missing_count']} 筆 ({data['missing_percentage']:.2f}%)\n")
            summary_text_io.write("\n")
        if summary_outlier_items:
            summary_has_issues = True
            summary_text_io.write("⚠️ 部分參數可能存在潛在異常值 (IQR 方法檢測)！\n")
            for param, data in summary_outlier_items.items():
                param_zh_txt = PARAMETER_INFO.get(param, {}).get('display_zh', param)
                percentage_txt = (data['outlier_iqr_count'] / data['valid_count']) * 100 if data['valid_count'] > 0 else 0
                summary_text_io.write(f"- {param_zh_txt}: 檢測到 {data['outlier_iqr_count']} 個潛在異常值 ({percentage_txt:.2f}%)\n")
            summary_text_io.write("\n")
        if not summary_has_issues:
            summary_text_io.write("✅ 數據品質良好！未檢測到顯著缺失或異常數據。\n\n")
        summary_text_io.write("----------------------------------------------------\n")
        summary_text_io.write(" 2. 詳細統計數據\n")
        summary_text_io.write("----------------------------------------------------\n\n")
        summary_text_io.write(stats_df.to_string(float_format="%.2f"))
        summary_text_io.write("\n\n====================================================\n")
        summary_txt_content = summary_text_io.getvalue()

        st.markdown("##### **單一檔案下載**")
        d_col1, d_col2, d_col3 = st.columns(3)
        with d_col1:
            st.download_button(label="📄 下載原始數據 (.csv)", data=raw_data_csv, file_name=f"raw_data_{current_station_name}_{time_range_str}.csv", mime="text/csv", use_container_width=True)
        with d_col2:
            st.download_button(label="📊 下載統計數據 (.csv)", data=stats_csv, file_name=f"statistics_{current_station_name}_{time_range_str}.csv", mime="text/csv", use_container_width=True)
        with d_col3:
            st.download_button(label="📝 下載文字摘要 (.txt)", data=summary_txt_content.encode('utf-8'), file_name=f"summary_report_{current_station_name}_{time_range_str}.txt", mime="text/plain", use_container_width=True)
        
        st.markdown("---")

        st.markdown("##### **組合包下載**")
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            has_charts = fig_pie_html or fig_box_html or fig_ts_html
            if has_charts:
                zip_buffer_html = io.BytesIO()
                with zipfile.ZipFile(zip_buffer_html, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
                    if fig_pie_html: zip_file.writestr(f"charts/quality_pie_chart_{current_station_name}_{time_range_str}.html", fig_pie_html)
                    if fig_box_html: zip_file.writestr(f"charts/boxplot_distribution_{current_station_name}_{time_range_str}.html", fig_box_html)
                    if fig_ts_html: zip_file.writestr(f"charts/timeseries_chart_{current_station_name}_{time_range_str}.html", fig_ts_html)
                st.download_button(label="📈 下載圖表包 (.zip)", data=zip_buffer_html.getvalue(), file_name=f"charts_package_{current_station_name}_{time_range_str}.zip", mime="application/zip", use_container_width=True)
            else:
                st.button("📈 無可下載圖表", disabled=True, use_container_width=True)
        with p_col2:
            zip_buffer_all = io.BytesIO()
            with zipfile.ZipFile(zip_buffer_all, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
                zip_file.writestr(f"data/raw_data_{current_station_name}_{time_range_str}.csv", raw_data_csv)
                zip_file.writestr(f"data/statistics_{current_station_name}_{time_range_str}.csv", stats_csv)
                if fig_pie_html: zip_file.writestr(f"charts/quality_pie_chart_{current_station_name}_{time_range_str}.html", fig_pie_html)
                if fig_box_html: zip_file.writestr(f"charts/boxplot_distribution_{current_station_name}_{time_range_str}.html", fig_box_html)
                if fig_ts_html: zip_file.writestr(f"charts/timeseries_chart_{current_station_name}_{time_range_str}.html", fig_ts_html)
                zip_file.writestr(f"summary_report_{current_station_name}_{time_range_str}.txt", summary_txt_content.encode('utf-8'))
            st.download_button(label="📥 一鍵打包所有產出 (.zip)", data=zip_buffer_all.getvalue(), file_name=f"analysis_package_{current_station_name}_{time_range_str}.zip", mime="application/zip", use_container_width=True)

    else:
        st.info("沒有數值型數據可供分析。")
else:
    st.info("請在上方選擇條件，然後點擊「🚀 產生統計報告」按鈕。")
