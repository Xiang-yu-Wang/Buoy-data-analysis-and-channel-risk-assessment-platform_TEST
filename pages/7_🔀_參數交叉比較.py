import streamlit as st
import pandas as pd
import plotly.express as px
from utils.helpers import get_station_name_from_id, load_year_data, PARAMETER_INFO, initialize_session_state
import io
from zipfile import ZipFile
from scipy.stats import linregress
import numpy as np
from datetime import datetime

st.title("🔀 參數交叉比較")
initialize_session_state()
st.write("在單一測站和特定時間範圍內，探索兩個不同物理參數之間的關聯性。")
st.markdown("---")

# --- 1. 從 session_state 讀取共享資料 (修改處) ---
locations = st.session_state.get('locations', [])
base_data_path = st.session_state.get('base_data_path', '')
# 讀取由主頁面提供的所有可用年份列表
all_available_years = st.session_state.get('available_years', [])

if not locations:
    st.warning("請返回主頁面以載入測站列表。")
    st.stop()

# 新增對年份列表的檢查
if not all_available_years:
    st.warning("沒有偵測到任何可用的年份資料，請檢查資料夾或返回主頁面重新載入。")
    st.stop()

# --- 輔助函式 (修改處) ---
@st.cache_data
def get_station_specific_years(station, years_to_check, data_path):
    """根據一個預先定義好的年份列表，檢查特定測站有哪些年份實際包含資料。"""
    valid_years = []
    for year in years_to_check:
        df = load_year_data(data_path, station, year)
        if df is not None and not df.empty:
            valid_years.append(year)
    return sorted(valid_years, reverse=True)

@st.cache_data
def calculate_data_quality(df):
    quality_stats = []
    total_records = len(df)
    params_to_check = df.select_dtypes(include=np.number).columns.tolist()

    for param in params_to_check:
        valid_count = df[param].count()
        missing_count = total_records - valid_count
        completeness = (valid_count / total_records * 100) if total_records > 0 else 0
        display_name = "未知參數"
        for key, info in PARAMETER_INFO.items():
            if key == param:
                display_name = info['display_zh']
                break

        quality_stats.append({
            "參數": display_name,
            "有效值": valid_count,
            "缺失值": missing_count,
            "完整度 (%)": completeness
        })
    return pd.DataFrame(quality_stats)

@st.cache_data
def detect_outliers(df):
    total_outlier_count = 0
    numeric_cols = df.select_dtypes(include=np.number).columns
    
    for col in numeric_cols:
        col_data = df[col].dropna()
        if col_data.empty:
            continue
            
        Q1 = col_data.quantile(0.25)
        Q3 = col_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
        total_outlier_count += len(outliers)
        
    return total_outlier_count

def render_quality_pie_chart(quality_df, outlier_count):
    if quality_df is None or quality_df.empty:
        return None

    total_valid_cells = quality_df['有效值'].sum()
    total_missing_cells = quality_df['缺失值'].sum()
    normal_valid_cells = total_valid_cells - outlier_count

    pie_data = {"類型": [], "數值": []}
    
    if normal_valid_cells > 0:
        pie_data["類型"].append("正常值")
        pie_data["數值"].append(normal_valid_cells)
    if outlier_count > 0:
        pie_data["類型"].append("異常值")
        pie_data["數值"].append(outlier_count)
    if total_missing_cells > 0:
        pie_data["類型"].append("缺失值")
        pie_data["數值"].append(total_missing_cells)

    if not pie_data["數值"]:
        return None

    pie_df = pd.DataFrame(pie_data)
    colors = {"正常值": "#1f77b4", "異常值": "#ff7f0e", "缺失值": "#d62728"}

    fig = px.pie(pie_df, values='數值', names='類型', title="全年整體數據品質",
                 color='類型', color_discrete_map=colors)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), legend_title_text='數據類型')
    return fig

# --- Session State 初始化與重設邏輯 ---
if 'analysis_run' not in st.session_state:
    st.session_state.analysis_run = False

def reset_analysis_state():
    st.session_state.analysis_run = False

# --- 2. 設定使用者輸入介面 (修改處) ---
st.sidebar.header("分析條件設定")
station = st.sidebar.selectbox("① 選擇測站", locations, key='pages_7_xc_station', on_change=reset_analysis_state, format_func=get_station_name_from_id)
station_name = get_station_name_from_id(station)

# 採用新的年份選擇邏輯
with st.sidebar:
    with st.spinner(f"正在查詢 {station} 的可用年份..."):
        station_specific_years = get_station_specific_years(station, all_available_years, base_data_path)

if not station_specific_years:
    st.sidebar.error(f"測站 '{station_name}' 找不到任何年份資料。")
    st.error(f"❌ 找不到測站 **{station_name}** 的任何年份資料，請嘗試選擇其他測站。")
    st.stop()

year = st.sidebar.selectbox("② 選擇年份", station_specific_years, key='pages_7_xc_year', on_change=reset_analysis_state)

param_options_display = {}
for col_name, info in PARAMETER_INFO.items():
    if info.get('type') == 'linear':
        param_options_display[f"{info['display_zh']} ({info['unit']})"] = col_name

st.sidebar.write("③ 選擇要比較的兩個參數：")
param_x_display = st.sidebar.selectbox("X 軸參數", list(param_options_display.keys()), key='pages_7_xc_param_x', on_change=reset_analysis_state)
param_y_display = st.sidebar.selectbox("Y 軸參數", list(param_options_display.keys()), index=1, key='pages_7_xc_param_y', on_change=reset_analysis_state)
param_x_col = param_options_display[param_x_display]
param_y_col = param_options_display[param_y_display]

if st.sidebar.button("🔬 進行交叉分析", use_container_width=True, type="primary"):
    if param_x_col == param_y_col:
        st.error("請選擇兩個不同的參數進行比較。")
    else:
        st.session_state.analysis_run = True
        
# --- 3. 執行與顯示分析結果 ---
if not st.session_state.analysis_run:
    st.info("👈🏻 請在左方側邊欄設定好分析條件，然後點擊「進行交叉分析」按鈕。")
    st.stop()

with st.spinner(f"正在載入 {station_name} 在 {year}年 的資料..."):
    df_year = load_year_data(base_data_path, station, year)

if df_year is None or df_year.empty:
    st.error(f"❌ 找不到 {station_name} 在 {year}年 的任何資料。")
    st.session_state.analysis_run = False
else:
    df_year['time'] = pd.to_datetime(df_year['time'])
    
    df_quality = calculate_data_quality(df_year)
    outlier_count = detect_outliers(df_year)
    numeric_cols = df_year.select_dtypes(include=np.number).columns
    df_desc = df_year[numeric_cols].describe().transpose()
    
    with st.expander(f"📊 資料儀表板：點此查看 {station_name} 在 {year} 年的數據概覽", expanded=True):
        col1, col2 = st.columns([0.6, 0.4])
        
        with col1:
            st.markdown("##### ① 數據統計概覽")
            
            sub_col1, sub_col2 = st.columns(2)
            if param_x_col in df_desc.index:
                with sub_col1:
                    st.markdown(f"**{PARAMETER_INFO.get(param_x_col, {}).get('display_zh', param_x_col)}**")
                    mean_val = df_desc.loc[param_x_col, 'mean']
                    max_val = df_desc.loc[param_x_col, 'max']
                    min_val = df_desc.loc[param_x_col, 'min']
                    st.metric(label="平均值", value=f"{mean_val:.2f}")
                    st.metric(label="最大值", value=f"{max_val:.2f}")
                    st.metric(label="最小值", value=f"{min_val:.2f}")

            if param_y_col in df_desc.index:
                with sub_col2:
                    st.markdown(f"**{PARAMETER_INFO.get(param_y_col, {}).get('display_zh', param_y_col)}**")
                    mean_val = df_desc.loc[param_y_col, 'mean']
                    max_val = df_desc.loc[param_y_col, 'max']
                    min_val = df_desc.loc[param_y_col, 'min']
                    st.metric(label="平均值", value=f"{mean_val:.2f}")
                    st.metric(label="最大值", value=f"{max_val:.2f}")
                    st.metric(label="最小值", value=f"{min_val:.2f}")

            st.markdown("---")
            st.caption("所有數值參數的詳細統計表")
            
            df_desc_view = df_desc[['count', 'mean', 'std', 'min', 'max']]
            st.dataframe(
                df_desc_view.style
                .background_gradient(cmap='viridis', subset=['mean', 'max'])
                .format("{:.2f}")
            )
        
        with col2:
            st.markdown("##### ② 數據品質報告")
            st.caption("此表計算了各參數的資料完整度。")
            st.dataframe(df_quality.style.background_gradient(cmap='Greens', subset=['完整度 (%)']).format({"完整度 (%)": "{:.2f}%"}))
            
            fig_pie = render_quality_pie_chart(df_quality, outlier_count)
            if fig_pie:
                pie_col_spacer1, pie_col_main, pie_col_spacer2 = st.columns([0.1, 0.8, 0.1])
                with pie_col_main:
                    st.plotly_chart(fig_pie, use_container_width=True)

            if outlier_count > 0:
                st.warning(f"⚠️ 數據中檢測到 {outlier_count} 個異常值 (使用 IQR 方法)。")
    st.markdown("---")

    cols_to_check = [c for c in [param_x_col, param_y_col] if c in df_year.columns]
    if len(cols_to_check) < 2:
        st.error(f"錯誤：資料中缺少所選參數。")
        st.stop()

    df_analysis = df_year[cols_to_check + ['time']].dropna()

    if df_analysis.empty or len(df_analysis) < 2:
        st.warning("在所選年份中，沒有足夠的共同數據點可進行交叉比較。")
    else:
        st.success(f"✅ 交叉分析完成！共找到 {len(df_analysis)} 筆可供比較的有效數據。")
        
        slope, intercept, r_value, _, _ = linregress(df_analysis[param_x_col], df_analysis[param_y_col])
        correlation = r_value
        r_squared = r_value**2
        equation_latex = fr"y = {slope:.4f}x {'+' if intercept >= 0 else ''} {intercept:.4f}"
        
        fig_scatter = px.scatter(
            df_analysis, x=param_x_col, y=param_y_col,
            labels={
                param_x_col: f"{PARAMETER_INFO.get(param_x_col, {}).get('display_zh', param_x_col)} ({PARAMETER_INFO.get(param_x_col, {}).get('unit', '')})",
                param_y_col: f"{PARAMETER_INFO.get(param_y_col, {}).get('display_zh', param_y_col)} ({PARAMETER_INFO.get(param_y_col, {}).get('unit', '')})"
            },
            trendline="ols", trendline_color_override="red",
            marginal_x="histogram", marginal_y="histogram",
            title="聯合分佈與趨勢線"
        )
        fig_scatter.update_layout(title_x=0.5)

        fig_timeseries = px.line(
            df_analysis, x='time', y=[param_x_col, param_y_col],
            labels={"value": "數值", "variable": "參數", "time": "時間"},
            title="參數時序變化圖"
        )
        fig_timeseries.update_layout(title_x=0.5)

        fig_density = px.density_heatmap(
            df_analysis, x=param_x_col, y=param_y_col,
            labels={
                param_x_col: f"{PARAMETER_INFO.get(param_x_col, {}).get('display_zh', param_x_col)} ({PARAMETER_INFO.get(param_x_col, {}).get('unit', '')})",
                param_y_col: f"{PARAMETER_INFO.get(param_y_col, {}).get('display_zh', param_y_col)} ({PARAMETER_INFO.get(param_y_col, {}).get('unit', '')})"
            },
            marginal_x="histogram", marginal_y="histogram",
            title="數據點密度分佈熱圖"
        )
        fig_density.update_layout(title_x=0.5)
        
        st.markdown(f"### 交叉分析結果：{station_name} ({year}年)")
        st.markdown(f"##### **{PARAMETER_INFO.get(param_x_col, {}).get('display_zh', param_x_col)}** vs. **{PARAMETER_INFO.get(param_y_col, {}).get('display_zh', param_y_col)}**")

        stat_col1, stat_col2, stat_col3 = st.columns(3)
        stat_col1.metric(label="皮爾森相關係數 (R)", value=f"{correlation:.4f}")
        stat_col2.metric(label="決定係數 (R-squared)", value=f"{r_squared:.4f}", help="R-squared 代表 Y 軸變異能被 X 軸解釋的百分比。")
        stat_col3.metric(label="共同數據筆數", value=f"{len(df_analysis)}")

        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 相關性散佈圖", "🕒 時序比較圖", "♨️ 數據密度圖", "🔢 詳細數據", "📥 下載專區"])
        
        with tab1:
            st.info("此圖顯示兩參數的直接關係。紅線為線性迴歸趨勢線，邊緣為各參數的數據分佈直方圖。")
            st.plotly_chart(fig_scatter, use_container_width=True)
            st.markdown("##### 迴歸分析結果")
            st.latex(f"{equation_latex} \\quad (R^2 = {r_squared:.4f})")
            
        with tab2:
            st.info("此圖將兩個參數的數值依時間繪製，可用於觀察兩者隨時間變化的同步性或延遲性。")
            st.plotly_chart(fig_timeseries, use_container_width=True)

        with tab3:
            st.info("此圖以顏色深淺表示數據點的密集程度，有助於識別數據集中的區域。")
            st.plotly_chart(fig_density, use_container_width=True)
        
        with tab4:
             st.subheader("分析所用數據")
             st.dataframe(df_analysis.rename(columns={
                 'time': '時間',
                 param_x_col: PARAMETER_INFO.get(param_x_col, {}).get('display_zh', param_x_col),
                 param_y_col: PARAMETER_INFO.get(param_y_col, {}).get('display_zh', param_y_col)
             }), use_container_width=True)

        with tab5:
            st.subheader("下載分析產出")
            csv_data = df_analysis.to_csv(index=False).encode('utf-8')
            html_buffer = io.StringIO()
            fig_scatter.write_html(html_buffer, include_plotlyjs='cdn')
            html_data = html_buffer.getvalue().encode('utf-8')
            quality_csv_data = df_quality.to_csv(index=False).encode('utf-8')

            summary_text = f"""
分析報告
=================================
測站: {station}
年份: {year}

數據品質概覽
---------------------------------
總記錄數: {len(df_year)}
檢測到的異常值總數 (IQR 方法): {outlier_count}

交叉分析參數
---------------------------------
X軸: {param_x_display}
Y軸: {param_y_display}
共同有效數據筆數: {len(df_analysis)}
皮爾森相關係數 (R): {correlation:.4f}
決定係數 (R-squared): {r_squared:.4f}
線性迴歸方程: {equation_latex.replace('y =', '').strip()}

=================================
全年數據品質報告
=================================
{df_quality.to_string()}

=================================
全年數據統計概覽
=================================
{df_desc.to_string()}
"""
            txt_data = summary_text.encode('utf-8')
            
            base_filename = f"analysis_{station_name}_{year}_{param_x_col}_vs_{param_y_col}"

            st.write("**個別檔案下載：**")
            dl_col1, dl_col2, dl_col3 = st.columns(3)
            with dl_col1:
                st.download_button("📄 下載交叉分析數據 (CSV)", csv_data, f"{base_filename}_data.csv", "text/csv", use_container_width=True, key="dl_csv")
            with dl_col2:
                st.download_button("📈 下載相關性圖表 (HTML)", html_data, f"{base_filename}_chart.html", "text/html", use_container_width=True, key="dl_html")
            with dl_col3:
                st.download_button("📝 下載完整文字報告 (TXT)", txt_data, f"{base_filename}_summary.txt", "text/plain", use_container_width=True, key="dl_txt")
            
            st.markdown("---") 

            zip_buffer = io.BytesIO()
            with ZipFile(zip_buffer, 'w') as zip_file:
                zip_file.writestr(f"{base_filename}_data.csv", csv_data)
                zip_file.writestr(f"{base_filename}_scatter_chart.html", html_data)
                zip_file.writestr(f"{base_filename}_summary.txt", txt_data)
                zip_file.writestr(f"quality_report_{station}_{year}.csv", quality_csv_data)
            
            st.download_button(
                label="📦 一鍵打包所有檔案 (ZIP)",
                data=zip_buffer.getvalue(),
                file_name=f"{base_filename}_package.zip",
                mime="application/zip",
                use_container_width=True,
                key="dl_zip"
            )
