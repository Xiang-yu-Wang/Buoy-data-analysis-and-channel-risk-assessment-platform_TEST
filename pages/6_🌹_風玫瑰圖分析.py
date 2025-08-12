import streamlit as st
import pandas as pd
import plotly.express as px
from utils.helpers import get_station_name_from_id, load_year_data, prepare_windrose_data, convert_df_to_csv, PARAMETER_INFO, load_single_file, initialize_session_state
import io
import zipfile
import os
import re

# 假設這些輔助函式存在於 utils/helpers.py 或其他地方
# @st.cache_data(ttl=3600)
# def load_year_data(base_path, station, year): ...
# @st.cache_data(ttl=3600)
# def prepare_windrose_data(df): ...
# def convert_df_to_csv(df): ...
# def load_single_file(path): ...
# PARAMETER_INFO = {"Wind_Speed": {"unit": "m/s"}}


# --- 快取與輔助函式 ---

@st.cache_data(ttl=3600)
def cached_load_year_data(base_path, station, year):
    """快取版本的 load_year_data"""
    return load_year_data(base_path, station, year)

@st.cache_data(ttl=3600)
def cached_prepare_windrose_data(df):
    """快取版本的 prepare_windrose_data"""
    return prepare_windrose_data(df)

#TODO: unify helper
@st.cache_data(ttl=3600)
def get_available_years_for_station(base_path, station):
    """掃描特定測站的資料目錄，找出所有包含數據的年份。"""
    data_path = os.path.join(base_path, station)
    if not data_path:
        return []
    years = set()
    year_month_pattern = re.compile(r"(\d{4})\d{2}\.csv")
    try:
        for filename in os.listdir(data_path):
            match = year_month_pattern.match(filename)
            if match:
                years.add(int(match.group(1)))
    except FileNotFoundError:
        return []
    return sorted(list(years))

@st.cache_data(ttl=3600)
def get_available_months_for_year(base_path, station, year):
    """對於給定的測站和年份，找出所有存在數據的月份。"""
    data_path = os.path.join(base_path, station)
    if not data_path:
        return []
    months = set()
    year_month_pattern = re.compile(r"(\d{4})(\d{2})\.csv")
    try:
        for filename in os.listdir(data_path):
            match = year_month_pattern.match(filename)
            if match and int(match.group(1)) == year:
                months.add(int(match.group(2)))
    except FileNotFoundError:
        return []
    return sorted(list(months))


# --- Streamlit App 主體 ---

st.markdown('<h1 style="color:white;">🌹 風玫瑰圖分析</h1>', unsafe_allow_html=True)
st.write("選擇一個測站及一個完整的時間區間，視覺化該時段的風向和風速分佈。")
st.markdown("---")
initialize_session_state()

# 初始化 session_state，用於儲存分析結果
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'animation_results' not in st.session_state:
    st.session_state.animation_results = None

# 建立一個回呼函式，當選項改變時清除舊結果
def clear_analysis_results():
    st.session_state.analysis_results = None
    st.session_state.animation_results = None

# 從 session_state 獲取必要資訊
locations = st.session_state.get('locations', [])
base_data_path = st.session_state.get('base_data_path', '')

if not locations or not base_data_path:
    st.warning("缺少必要的設定資訊，請返回主頁面載入測站列表並設定資料夾。")
    st.stop()

st.sidebar.header("分析設定")
# 為所有輸入元件加上 on_change 回呼函式
station = st.sidebar.selectbox(
    "選擇測站:",
    options=locations,
    key='pages_6_wr_station',
    on_change=clear_analysis_results,
    format_func=get_station_name_from_id
)

station_specific_years = get_available_years_for_station(base_data_path, station)

if not station_specific_years:
    st.error(f"錯誤：在測站 '{station}' 的路徑下找不到任何格式為 'YYYYMM.csv' 的數據檔案。")
    st.info("請檢查您的資料夾結構是否正確，且檔案名稱是否符合規範（例如：202301.csv）。")
    st.stop()

analysis_mode = st.sidebar.radio(
    "選擇分析模式:",
    ("單期分析", "逐月動畫"),
    key='pages_6_wr_analysis_mode',
    on_change=clear_analysis_results
)

# --- 模式一: 單期分析 (包含儀表板) ---
if analysis_mode == "單期分析":
    st.subheader("設定分析區間 (單期分析)")

    is_ready_to_plot = True

    col1, col2 = st.columns(2)
    with col1:
        st.write("#### 開始日期")
        start_year = st.selectbox("年份", station_specific_years, key='pages_6_wr_start_year', on_change=clear_analysis_results)
        available_start_months = get_available_months_for_year(base_data_path, station, start_year)
        if not available_start_months:
            st.warning(f"在 {start_year} 年找不到任何月份的資料。")
            is_ready_to_plot = False
            start_month = None
        else:
            start_month = st.selectbox("月份", available_start_months, key='pages_6_wr_start_month', on_change=clear_analysis_results)

    with col2:
        st.write("#### 結束日期")
        default_end_year_index = len(station_specific_years) - 1
        end_year = st.selectbox("年份", station_specific_years, index=default_end_year_index, key='pages_6_wr_end_year', on_change=clear_analysis_results)
        available_end_months = get_available_months_for_year(base_data_path, station, end_year)
        if not available_end_months:
            st.warning(f"在 {end_year} 年找不到任何月份的資料。")
            is_ready_to_plot = False
            end_month = None
        else:
            default_end_month_index = len(available_end_months) - 1
            end_month = st.selectbox("月份", available_end_months, index=default_end_month_index, key='pages_6_wr_end_month', on_change=clear_analysis_results)

    if is_ready_to_plot and st.button("🌹 產生風玫瑰圖", key='pages_6_wr_button_single', use_container_width=True):
        if start_year > end_year or (start_year == end_year and start_month > end_month):
            st.error("錯誤：開始日期不能晚於結束日期。")
        else:
            with st.spinner(f"正在為 {get_station_name_from_id(station)} 載入 {start_year} 年至 {end_year} 年的資料..."):
                all_dfs = [cached_load_year_data(base_data_path, station, year) for year in range(start_year, end_year + 1)]
                all_dfs = [df for df in all_dfs if df is not None]
                if not all_dfs:
                    st.error(f"在 {start_year} 年至 {end_year} 年的範圍內找不到 {station} 的任何資料。")
                    st.stop()
                combined_df = pd.concat(all_dfs, ignore_index=True)

            start_date = pd.to_datetime(f'{start_year}-{start_month:02d}-01')
            end_date = pd.to_datetime(f'{end_year}-{end_month:02d}-01') + pd.offsets.MonthEnd(0)
            data_to_plot = combined_df[(combined_df['time'] >= start_date) & (combined_df['time'] <= end_date)]
            
            if data_to_plot.empty:
                st.error(f"在指定的區間內找不到任何資料可供分析。")
                st.session_state.analysis_results = None
            else:
                windrose_df = cached_prepare_windrose_data(data_to_plot)
                if windrose_df is None:
                     st.warning(f"在指定區間內，{get_station_name_from_id(station)} 雖然有資料，但缺乏有效的風速或風向數據。")
                     st.session_state.analysis_results = None
                else:
                    st.session_state.analysis_results = {
                        "station": get_station_name_from_id(station),
                        "start_year": start_year,
                        "start_month": start_month,
                        "end_year": end_year,
                        "end_month": end_month,
                        "start_date": start_date,
                        "end_date": end_date,
                        "data_to_plot": data_to_plot,
                        "windrose_df": windrose_df
                    }

if st.session_state.analysis_results:
    results = st.session_state.analysis_results
    station = results["station"]
    title_time_range = f'{results["start_year"]}年{results["start_month"]}月至{results["end_year"]}年{results["end_month"]}月'
    
    st.subheader(f"{station} - {title_time_range} 風玫瑰圖")

    wind_speed_unit = PARAMETER_INFO.get("Wind_Speed", {}).get("unit", "m/s")
    speed_labels = ['0-2 m/s', '2-4 m/s', '4-6 m/s', '6-8 m/s', '8-10 m/s', '10-12 m/s', f'>12 {wind_speed_unit}']
    fig = px.bar_polar(results["windrose_df"], r="percentage", theta="direction_bin", color="speed_bin",
                       color_discrete_sequence=px.colors.sequential.Plasma_r,
                       category_orders={"speed_bin": speed_labels},
                       hover_data={"percentage": ":.2f%", "frequency": True})
    fig.update_layout(title=f'{station} - {title_time_range} 風玫瑰圖',
                      legend_title=f'風速 ({wind_speed_unit})',
                      polar_angularaxis_rotation=90, polar_angularaxis_direction='clockwise', font=dict(color="black"))

    tab1, tab2, tab3 = st.tabs(["📊 圖表", "📄 圖表數據", "📈 儀表板"])

    with tab1:
        st.plotly_chart(fig, use_container_width=True)
    with tab2:
        st.dataframe(results["windrose_df"])
    with tab3:
        st.subheader("數據品質儀表板")
        st.markdown("#### 📖 數據概覽")
        expected_interval = pd.Timedelta(minutes=10)
        total_duration = results["end_date"] - results["start_date"]
        expected_points = total_duration / expected_interval if total_duration > pd.Timedelta(0) else 0
        total_points = len(results["data_to_plot"])
        completeness = (total_points / expected_points) * 100 if expected_points > 0 else 0
        dash_col1, dash_col2, dash_col3 = st.columns(3)
        dash_col1.metric("時間起點", results["start_date"].strftime('%Y-%m-%d'))
        dash_col2.metric("時間終點", results["end_date"].strftime('%Y-%m-%d'))
        dash_col3.metric("資料完整度", f"{completeness:.2f}%", help=f"此為與理論上應有資料筆數（每10分鐘一筆）的比對結果。")
        st.markdown("#### 🔍 數據品質概覽")
        valid_wind_data = results["data_to_plot"].dropna(subset=['Wind_Speed', 'Wind_Direction'])
        valid_count = len(valid_wind_data)
        missing_count = total_points - valid_count
        pie_data = pd.DataFrame({'類別': ['有效風數據', '無效/缺失風數據'], '筆數': [valid_count, missing_count]})
        fig_pie = px.pie(pie_data, values='筆數', names='類別', title='風數據品質分佈',
                         color_discrete_sequence=['#1f77b4', '#d62728'], hole=0.3)
        fig_pie.update_traces(textinfo='percent+label', pull=[0, 0.05])
        fig_pie.update_layout(legend_title_text='數據類別', font=dict(color="black"))
        dash_col4, dash_col5 = st.columns([0.6, 0.4])
        with dash_col4: st.plotly_chart(fig_pie, use_container_width=True)
        with dash_col5:
            st.metric("總資料筆數", f"{total_points:,}")
            st.metric("有效風數據筆數", f"{valid_count:,}")
            st.metric("無效/缺失筆數", f"{missing_count:,}")
            st.info("有效風數據指「風速」和「風向」欄位皆有數值的資料點。")

    with st.expander("📦 點此展開/收合下載選項"):
        raw_data_csv = convert_df_to_csv(results["data_to_plot"])
        windrose_table_csv = convert_df_to_csv(results["windrose_df"])
        fig_html = fig.to_html()
        dashboard_report_str = f"""
# 風玫瑰圖分析報告 - 儀表板
## 測站資訊
- 測站名稱: {station}
- 分析區間: {results["start_date"].strftime('%Y-%m-%d')} 至 {results["end_date"].strftime('%Y-%m-%d')}
## 數據概覽
- 資料完整度: {completeness:.2f}%
  - (基於每 10 分鐘一筆的理論數據量)
- 期間內理論應有筆數: {int(expected_points):,}
- 實際載入筆數: {total_points:,}
## 數據品質概覽 (針對風速與風向)
- 總資料筆數: {total_points:,}
- 有效風數據筆數: {valid_count:,}
- 無效/缺失風數據筆數: {missing_count:,}
- 有效數據比例: {(valid_count / total_points * 100) if total_points > 0 else 0:.2f}%
---
報告生成時間: {pd.Timestamp.now('Asia/Taipei').strftime('%Y-%m-%d %H:%M:%S')}
"""
        dl_col1, dl_col2 = st.columns(2)
        s_y, s_m, e_y, e_m = results["start_year"], results["start_month"], results["end_year"], results["end_month"]
        with dl_col1:
            st.download_button("📥 區間資料 (CSV)", raw_data_csv, f"raw_data_{station}_{s_y}{s_m:02d}-{e_y}{e_m:02d}.csv", "text/csv", use_container_width=True)
        with dl_col2:
            st.download_button("📥 圖表數據 (CSV)", windrose_table_csv, f"windrose_data_{station}_{s_y}{s_m:02d}-{e_y}{e_m:02d}.csv", "text/csv", use_container_width=True)
        dl_col3, dl_col4 = st.columns(2)
        with dl_col3:
            st.download_button("📥 互動圖表 (HTML)", fig_html, f"windrose_chart_{station}_{s_y}{s_m:02d}-{e_y}{e_m:02d}.html", "text/html", use_container_width=True)
        with dl_col4:
            st.download_button("📥 文字報告 (TXT)", dashboard_report_str.encode('utf-8'), f"dashboard_report_{station}_{s_y}{s_m:02d}-{e_y}{e_m:02d}.txt", "text/plain", use_container_width=True)
        st.markdown("---")
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            zip_file.writestr(f"raw_data_{station}.csv", raw_data_csv)
            zip_file.writestr(f"windrose_data_{station}.csv", windrose_table_csv)
            zip_file.writestr(f"windrose_chart_{station}.html", fig_html)
            zip_file.writestr(f"dashboard_report_{station}.txt", dashboard_report_str.encode('utf-8'))
        st.download_button("📥 一鍵打包下載 (.zip)", zip_buffer.getvalue(), f"windrose_package_{station}_{s_y}{s_m:02d}-{e_y}{e_m:02d}.zip", "application/zip", use_container_width=True)


# --- 模式二: 逐月動畫 ---
elif analysis_mode == "逐月動畫":
    st.subheader("設定分析年份 (逐月動畫)")
    selected_anim_year = st.selectbox("選擇年份:", station_specific_years, key='pages_6_wr_anim_year', on_change=clear_analysis_results)

    if st.button("🎬 產生逐月動畫", key='pages_6_wr_button_anim', use_container_width=True):
        data_source_path = find_station_data_path(base_data_path, station)
        if data_source_path is None:
            st.error(f"錯誤：在測站 '{station}' 的路徑下未找到有效的數據來源資料夾。")
            st.session_state.animation_results = None
        else:
            with st.spinner(f"正在為 {station} 載入 {selected_anim_year} 年的逐月資料..."):
                all_monthly_windrose_dfs = []
                all_raw_monthly_dfs = [] # <<< 新增: 用於儀表板的原始數據列表
                max_percentage = 0
                months_to_process = get_available_months_for_year(base_data_path, station, selected_anim_year)
                
                if not months_to_process:
                    st.error(f"在 {selected_anim_year} 年， '{station}' 沒有任何月份的數據可供處理。")
                    st.session_state.animation_results = None
                else:
                    for month_num in months_to_process:
                        file_path = os.path.join(data_source_path, f"{selected_anim_year}{month_num:02d}.csv")
                        if os.path.exists(file_path):
                            df_month = load_single_file(file_path)
                            if df_month is not None and not df_month.empty:
                                all_raw_monthly_dfs.append(df_month) # <<< 新增: 收集原始月數據
                                windrose_df_month = cached_prepare_windrose_data(df_month)
                                if windrose_df_month is not None and not windrose_df_month.empty:
                                    windrose_df_month['month_label'] = f"{selected_anim_year}年{month_num:02d}月"
                                    all_monthly_windrose_dfs.append(windrose_df_month)
                                    current_max_pct = windrose_df_month['percentage'].max()
                                    if current_max_pct > max_percentage:
                                        max_percentage = current_max_pct
                    
                    if not all_monthly_windrose_dfs:
                        st.error(f"錯誤：在 {selected_anim_year} 年，所有月份都找不到有效風速或風向數據，無法生成動畫。")
                        st.session_state.animation_results = None
                    else:
                        # <<< 新增: 將儀表板數據也存入 session_state >>>
                        yearly_df = pd.concat(all_raw_monthly_dfs, ignore_index=True)
                        st.session_state.animation_results = {
                            "station": station,
                            "year": selected_anim_year,
                            "df": pd.concat(all_monthly_windrose_dfs, ignore_index=True),
                            "max_percentage": max_percentage,
                            "months_order": [f"{selected_anim_year}年{m:02d}月" for m in months_to_process],
                            "yearly_df": yearly_df # <<< 新增
                        }

    if st.session_state.animation_results:
        res = st.session_state.animation_results
        station = res["station"]
        year = res["year"]
        df = res["df"]
        
        df['month_label'] = pd.Categorical(df['month_label'], categories=res["months_order"], ordered=True)
        df = df.sort_values('month_label')
        
        st.subheader(f"{station} - {year} 年逐月風玫瑰動畫")
        
        wind_speed_unit = PARAMETER_INFO.get("Wind_Speed", {}).get("unit", "m/s")
        speed_labels = ['0-2 m/s', '2-4 m/s', '4-6 m/s', '6-8 m/s', '8-10 m/s', '10-12 m/s', f'>12 {wind_speed_unit}']
        direction_labels = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
        fig_anim = px.bar_polar(
            df, r="percentage", theta="direction_bin", color="speed_bin",
            color_discrete_sequence=px.colors.sequential.Plasma_r,
            category_orders={"speed_bin": speed_labels, "direction_bin": direction_labels},
            hover_data={"percentage": ":.2f%", "frequency": True},
            animation_frame="month_label", animation_group="direction_bin",
            range_r=[0, res["max_percentage"] * 1.1])
        fig_anim.update_layout(
            title=f'{station} - {year} 年逐月風玫瑰圖', legend_title=f'風速 ({wind_speed_unit})',
            polar_angularaxis_rotation=90, polar_angularaxis_direction='clockwise', font=dict(color="white"))
        if fig_anim.layout.updatemenus:
            fig_anim.layout.updatemenus[0].font.color = 'white'
            fig_anim.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 500
            fig_anim.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 300
        if fig_anim.layout.sliders:
            fig_anim.layout.sliders[0].font.color = 'white'
            fig_anim.layout.sliders[0].currentvalue.font.color = 'white'
        
        # <<< 修改: 新增儀表板分頁 >>>
        tab1_anim, tab2_anim, tab3_anim = st.tabs(["📊 動畫圖表", "📄 動畫數據", "📈 全年儀表板"])
        
        with tab1_anim:
            st.plotly_chart(fig_anim, use_container_width=True)
        with tab2_anim:
            st.dataframe(df)
        # <<< 新增: 全年儀表板的顯示邏輯 >>>
        with tab3_anim:
            yearly_df = res["yearly_df"]
            st.subheader(f"{year} 全年度數據品質儀表板")
            st.markdown("#### 📖 數據概覽")
            start_date = pd.to_datetime(f'{year}-01-01')
            end_date = pd.to_datetime(f'{year}-12-31')
            expected_interval = pd.Timedelta(minutes=10)
            total_duration = end_date - start_date
            expected_points = total_duration / expected_interval if total_duration > pd.Timedelta(0) else 0
            total_points = len(yearly_df)
            completeness = (total_points / expected_points) * 100 if expected_points > 0 else 0
            dash_col1, dash_col2, dash_col3 = st.columns(3)
            dash_col1.metric("分析年份", f"{year}年")
            dash_col2.metric("總資料筆數", f"{total_points:,}")
            dash_col3.metric("全年資料完整度", f"{completeness:.2f}%", help=f"此為與理論上應有資料筆數（每10分鐘一筆）的比對結果。")
            st.markdown("#### 🔍 數據品質概覽")
            valid_wind_data = yearly_df.dropna(subset=['Wind_Speed', 'Wind_Direction'])
            valid_count = len(valid_wind_data)
            missing_count = total_points - valid_count
            pie_data = pd.DataFrame({'類別': ['有效風數據', '無效/缺失風數據'], '筆數': [valid_count, missing_count]})
            fig_pie_anim = px.pie(pie_data, values='筆數', names='類別', title='全年風數據品質分佈',
                             color_discrete_sequence=['#1f77b4', '#d62728'], hole=0.3)
            fig_pie_anim.update_traces(textinfo='percent+label', pull=[0, 0.05])
            fig_pie_anim.update_layout(legend_title_text='數據類別', font=dict(color="black"))
            dash_col4, dash_col5 = st.columns([0.6, 0.4])
            with dash_col4: st.plotly_chart(fig_pie_anim, use_container_width=True)
            with dash_col5:
                st.metric("有效風數據筆數", f"{valid_count:,}")
                st.metric("無效/缺失筆數", f"{missing_count:,}")
                st.info("有效風數據指「風速」和「風向」欄位皆有數值的資料點。")

        with st.expander("📦 點此展開/收合下載選項"):
            csv_anim_data = convert_df_to_csv(df)
            html_anim_chart = fig_anim.to_html(full_html=False, include_plotlyjs='cdn')
            # <<< 新增: 全年儀表板報告 >>>
            dashboard_report_str_anim = f"""
# 逐月動畫分析報告 - 全年儀表板
## 測站資訊
- 測站名稱: {station}
- 分析年份: {year}
## 全年數據概覽
- 資料完整度: {completeness:.2f}%
  - (基於每 10 分鐘一筆的理論數據量)
- 全年理論應有筆數: {int(expected_points):,}
- 全年實際載入筆數: {total_points:,}
## 全年數據品質概覽 (針對風速與風向)
- 總資料筆數: {total_points:,}
- 有效風數據筆數: {valid_count:,}
- 無效/缺失風數據筆數: {missing_count:,}
- 有效數據比例: {(valid_count / total_points * 100) if total_points > 0 else 0:.2f}%
---
報告生成時間: {pd.Timestamp.now('Asia/Taipei').strftime('%Y-%m-%d %H:%M:%S')}
"""
            # <<< 修改: 調整下載按鈕佈局 >>>
            dl_col_anim_1, dl_col_anim_2 = st.columns(2)
            with dl_col_anim_1:
                st.download_button(label="📥 動畫數據 (CSV)", data=csv_anim_data, file_name=f"animated_windrose_data_{station}_{year}.csv", mime="text/csv", use_container_width=True)
            with dl_col_anim_2:
                st.download_button(label="📥 動畫圖表 (HTML)", data=html_anim_chart.encode('utf-8'), file_name=f"animated_windrose_chart_{station}_{year}.html", mime="text/html", use_container_width=True)
            
            st.download_button(label="📥 全年儀表板報告 (TXT)", data=dashboard_report_str_anim.encode('utf-8'), file_name=f"dashboard_report_{station}_{year}.txt", mime="text/plain", use_container_width=True)
            
            st.markdown("---")
            zip_buffer_anim = io.BytesIO()
            with zipfile.ZipFile(zip_buffer_anim, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
                zip_file.writestr(f"animated_windrose_data_{station}_{year}.csv", csv_anim_data)
                zip_file.writestr(f"animated_windrose_chart_{station}_{year}.html", html_anim_chart.encode('utf-8'))
                zip_file.writestr(f"dashboard_report_{station}_{year}.txt", dashboard_report_str_anim.encode('utf-8')) # <<< 新增
            st.download_button(label="📥 一鍵打包下載 (.zip)", data=zip_buffer_anim.getvalue(), file_name=f"animated_windrose_package_{station}_{year}.zip", mime="application/zip", use_container_width=True)
