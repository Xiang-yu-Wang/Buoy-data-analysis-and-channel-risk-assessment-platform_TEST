import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
import zipfile
from scipy import signal
from scipy.stats import mstats, linregress

from utils.helpers import get_station_name_from_id, initialize_session_state

# 為了讓此腳本能獨立運行，我們模擬輔助函式的功能
# 在您的專案中，請確保 from utils.helpers import ... 是有效的
def load_year_data(base_path, station, year):
    """模擬從檔案載入資料的函式。"""
    # 為了測試篩選功能，讓 StationC 在 2021 年沒有資料
    if station == 'StationC' and year == '2021':
        return None
    
    np.random.seed(hash(f"{station}{year}") % (2**32 - 1)) # 確保每次生成的假資料都一樣
    date_rng = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq='H')
    df = pd.DataFrame(date_rng, columns=['time'])
    df['Wave_Height_Significant'] = np.random.normal(1.5, 0.5, size=len(date_rng))
    df['Wave_Mean_Period'] = np.random.normal(8, 1, size=len(date_rng))
    df['Wave_Peak_Period'] = np.random.normal(12, 2, size=len(date_rng))
    df['Wind_Speed'] = np.random.normal(10, 3, size=len(date_rng))
    df['Wind_Gust_Speed'] = np.random.normal(15, 4, size=len(date_rng))
    df['Air_Temperature'] = np.random.normal(25, 5, size=len(date_rng))
    df['Sea_Temperature'] = np.random.normal(26, 3, size=len(date_rng))
    df['Air_Pressure'] = np.random.normal(1010, 5, size=len(date_rng))
    df['Wind_Direction'] = np.random.randint(0, 360, size=len(date_rng))
    df['Wave_Main_Direction'] = np.random.randint(0, 360, size=len(date_rng))
    # 隨機插入一些缺失值與異常值
    df.loc[df.sample(frac=0.1).index, 'Wave_Height_Significant'] = np.nan
    df.loc[df.sample(frac=0.02).index, 'Wind_Speed'] = 99.9 # 異常值
    df.loc[df.sample(frac=0.01).index, 'Wave_Mean_Period'] = -10 # 異常值
    return df

def convert_df_to_csv(df):
    """將 DataFrame 轉換為 CSV 格式的 bytes。"""
    return df.to_csv(index=False).encode('utf-8')


# --- 頁面基礎設定 ---
st.set_page_config(layout="wide")
st.title('📈 測站資料分析平台')
st.write("提供測站相關性分析與數據品質檢視功能。")
initialize_session_state()

# --- 常數與字典定義 ---
PARAM_DISPLAY_NAMES = {
    "Wave_Height_Significant": "示性波高", "Wave_Mean_Period": "平均波週期",
    "Wave_Peak_Period": "波浪尖峰週期", "Wind_Speed": "風速",
    "Wind_Gust_Speed": "陣風風速", "Air_Temperature": "氣溫",
    "Sea_Temperature": "海面溫度", "Air_Pressure": "氣壓",
    "Wind_Direction": "風向", "Wave_Main_Direction": "波向"
}
PARAM_UNITS = {
    "Wave_Height_Significant": " (m)", "Wave_Mean_Period": " (sec)",
    "Wave_Peak_Period": " (sec)", "Wind_Speed": " (m/s)",
    "Wind_Gust_Speed": " (m/s)", "Air_Temperature": " (°C)",
    "Sea_Temperature": " (°C)", "Air_Pressure": " (hPa)",
    "Wind_Direction": " (°)", "Wave_Main_Direction": " (°)"
}

# --- 快取計算函式 (核心優化) ---
@st.cache_data
def calculate_data_quality(df):
    """
    計算傳入的 DataFrame 的數據品質，包含異常值檢測 (IQR)。
    """
    if df is None or df.empty:
        return None

    quality_stats = []
    params_to_check = [col for col in df.columns if col != 'time']
    total_records = len(df)

    for param in params_to_check:
        valid_series = df[param].dropna()
        valid_count = len(valid_series)
        missing_count = total_records - valid_count
        completeness = (valid_count / total_records * 100) if total_records > 0 else 0
        
        outlier_count = 0
        if valid_count > 1 and pd.api.types.is_numeric_dtype(valid_series):
            Q1 = valid_series.quantile(0.25)
            Q3 = valid_series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = valid_series[(valid_series < lower_bound) | (valid_series > upper_bound)]
            outlier_count = len(outliers)

        quality_stats.append({
            "參數": PARAM_DISPLAY_NAMES.get(param, param),
            "有效值": valid_count,
            "缺失值": missing_count,
            "異常值": outlier_count,
            "完整度 (%)": completeness
        })
    
    return pd.DataFrame(quality_stats)

@st.cache_data
def calculate_single_year_correlation(_base_path, station1, station2, year, param_col, analysis_type):
    """
    載入並計算單一年份的相關性資料，並包含兩測站的數據品質報告。
    """
    df1_raw = load_year_data(_base_path, station1, year)
    df2_raw = load_year_data(_base_path, station2, year)
    
    quality1_df = calculate_data_quality(df1_raw)
    quality2_df = calculate_data_quality(df2_raw)

    results = {"df1_raw": df1_raw, "df2_raw": df2_raw, "quality1": quality1_df, "quality2": quality2_df}

    if df1_raw is None or df2_raw is None or df1_raw.empty or df2_raw.empty:
        return {"error": f"錯誤：未能載入 {year} 年 {station1} 或 {station2} 的資料，或資料為空。", **results}

    if analysis_type == 'linear':
        if param_col not in df1_raw.columns or param_col not in df2_raw.columns:
            return {"error": f"錯誤：資料中缺少 '{PARAM_DISPLAY_NAMES.get(param_col, param_col)}' 欄位。", **results}
        
        merged = pd.merge(df1_raw[['time', param_col]], df2_raw[['time', param_col]], on='time', how='inner', suffixes=(f'_{station1}', f'_{station2}')).dropna()
        if len(merged) < 2:
            return {"error": "共同時間內無足夠數據進行相關性分析。", **results}
            
        slope, intercept, r_value, _, _ = linregress(merged.iloc[:, 1], merged.iloc[:, 2])
        results.update({
            "type": "linear", "merged": merged, "corr": r_value, "slope": slope, 
            "intercept": intercept, "r_squared": r_value**2, "x_col": merged.columns[1], "y_col": merged.columns[2]
        })

    elif analysis_type == 'circular':
        mag_col = 'Wind_Speed' if 'Wind' in param_col else 'Wave_Height_Significant'
        if not all(c in df1_raw.columns and c in df2_raw.columns for c in [param_col, mag_col]):
            return {"error": f"錯誤：資料中缺少 '{PARAM_DISPLAY_NAMES.get(param_col, param_col)}' 或對應量值欄位。", **results}

        for df, s_name in [(df1_raw, station1), (df2_raw, station2)]:
            df_dropna = df.dropna(subset=[param_col, mag_col])
            if not df_dropna.empty:
                rad = np.radians(df_dropna[param_col])
                mag = df_dropna[mag_col]
                df.loc[df_dropna.index, f'u_{s_name}'] = -mag * np.sin(rad)
                df.loc[df_dropna.index, f'v_{s_name}'] = -mag * np.cos(rad)

        merged_uv = pd.merge(df1_raw[['time', f'u_{station1}', f'v_{station1}']], df2_raw[['time', f'u_{station2}', f'v_{station2}']], on='time', how='inner').dropna()

        if len(merged_uv) < 2:
            return {"error": "共同時間內無足夠U/V分量數據進行分析。", **results}

        corr_u = merged_uv[f'u_{station1}'].corr(merged_uv[f'u_{station2}'])
        corr_v = merged_uv[f'v_{station1}'].corr(merged_uv[f'v_{station2}'])
        results.update({
            "type": "circular", "merged": merged_uv, "corr_u": corr_u, "corr_v": corr_v, "mag_col": mag_col
        })
        
    return results

@st.cache_data
def calculate_yearly_trend(_base_path, s1, s2, param_col, start_y, end_y):
    """
    計算逐年相關性趨勢，並回傳用於計算的資料點數量。
    """
    results_data = []
    years_to_analyze = range(int(start_y), int(end_y) + 1)
    bar = st.progress(0, "準備開始...")
    for i, year_val in enumerate(years_to_analyze):
        bar.progress((i + 1) / len(years_to_analyze), f"正在處理 {year_val} 年...")
        df1 = load_year_data(_base_path, s1, str(year_val))
        df2 = load_year_data(_base_path, s2, str(year_val))
        
        corr = np.nan
        pair_count = 0
        
        if df1 is not None and not df1.empty and df2 is not None and not df2.empty and param_col in df1.columns and param_col in df2.columns:
            merged = pd.merge(df1[['time', param_col]], df2[['time', param_col]], on='time', how='inner').dropna()
            pair_count = len(merged)
            if pair_count > 1:
                corr = merged.iloc[:, 1].corr(merged.iloc[:, 2])
                
        results_data.append({'年份': year_val, '相關係數': corr, '配對資料點數': pair_count})
    bar.empty()
    
    results_df = pd.DataFrame(results_data)
    return results_df

@st.cache_data
def get_common_available_years(_base_path, station1, station2, all_years):
    """
    查詢兩個指定測站共同擁有資料的年份列表。
    """
    common_years = []
    for year in all_years:
        df1_check = load_year_data(_base_path, station1, year)
        if df1_check is not None and not df1_check.empty:
            df2_check = load_year_data(_base_path, station2, year)
            if df2_check is not None and not df2_check.empty:
                common_years.append(year)
    return sorted(common_years, reverse=True)

# --- 繪圖與UI渲染輔助函式 ---
def create_download_package(files_dict):
    """
    將多個檔案內容打包成一個 ZIP 檔。
    """
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for filename, content in files_dict.items():
            if isinstance(content, str):
                zf.writestr(filename, content.encode('utf-8'))
            else:
                zf.writestr(filename, content)
    return zip_buffer.getvalue()

def render_quality_pie_chart(quality_df, title):
    """根據品質摘要DataFrame，渲染包含異常值的整體數據品質圓餅圖"""
    if quality_df is None or quality_df.empty or '異常值' not in quality_df.columns:
        st.warning("無品質資料可繪圖，或資料格式不含異常值資訊。")
        return
        
    total_valid = quality_df['有效值'].sum()
    total_missing = quality_df['缺失值'].sum()
    total_outliers = quality_df['異常值'].sum()
    total_normal = max(0, total_valid - total_outliers)
    
    if total_normal + total_outliers + total_missing == 0:
        st.warning("沒有數據可供繪製圓餅圖。")
        return

    pie_data = {
        '類別': ['正常數據', '潛在異常值', '缺失數據'],
        '數量': [total_normal, total_outliers, total_missing]
    }
    pie_df = pd.DataFrame(pie_data)
    pie_df = pie_df[pie_df['數量'] > 0]
    
    if pie_df.empty:
        st.info("所有數據皆完整且無異常值。")
        return

    fig = px.pie(
        pie_df, 
        values='數量', 
        names='類別', 
        title=title,
        hole=0.4,
        color_discrete_map={
            '正常數據': '#1f77b4',
            '潛在異常值': '#ff7f0e',
            '缺失數據': '#d62728'
        }
    )
    fig.update_traces(
        textposition='inside', 
        textinfo='percent+label', 
        pull=[0.05 if cat in ['潛在異常值', '缺失數據'] else 0 for cat in pie_df['類別']]
    )
    fig.update_layout(
        showlegend=True,
        margin=dict(l=0, r=0, t=40, b=0),
        legend_title_text='數據類別'
    )
    st.plotly_chart(fig, use_container_width=True)

def create_rose_plot(df, station_name, param_name_display, dir_col, mag_col):
    """根據方向和量值數據，建立一個極座標的玫瑰圖。"""
    df_plot = df[[dir_col, mag_col]].dropna()
    if df_plot.empty: return None
    bins = np.arange(-11.25, 360, 22.5)
    labels = ["北", "北北東", "東北", "東北東", "東", "東南東", "東南", "南南東", "南", "南南西", "西南", "西西南", "西", "西北西", "西北", "北北西"]
    dir_binned_series = pd.cut(df_plot[dir_col] % 360, bins=bins, labels=labels, right=False)
    dir_freq = dir_binned_series.value_counts().reindex(labels, fill_value=0)
    fig = px.bar_polar(
        dir_freq, r=dir_freq.values, theta=dir_freq.index,
        title=f"{station_name} - {param_name_display}分佈玫瑰圖", template="seaborn",
        labels={'theta': '方向', 'r': '頻率 (觀測次數)'}, color_discrete_sequence=px.colors.sequential.Plasma_r
    )
    fig.update_layout(polar=dict(angularaxis=dict(direction="clockwise", period=360)))
    return fig

def render_advanced_analysis(merged, x_col, y_col, station1, station2):
    """渲染進階分析圖表 (Cross-correlation, Q-Q Plot)"""
    adv_col1, adv_col2 = st.columns(2)
    x_series, y_series = merged[x_col], merged[y_col]
    
    with adv_col1:
        st.subheader("時間延遲相關圖 (Cross-correlation)")
        st.caption(f"分析 {station1} 的訊號移動多少小時後，會與 {station2} 的訊號最相關。")
        max_lag_hours = 48
        if len(x_series) > max_lag_hours:
            x_norm = (x_series - x_series.mean()) / x_series.std()
            y_norm = (y_series - y_series.mean()) / y_series.std()
            correlation = signal.correlate(x_norm, y_norm, mode='full') / len(x_norm)
            lags = signal.correlation_lags(len(x_norm), len(y_norm), mode="full")
            
            lag_filter = (lags >= -max_lag_hours) & (lags <= max_lag_hours)
            lags, correlation = lags[lag_filter], correlation[lag_filter]

            best_lag = lags[np.argmax(np.abs(correlation))]
            best_corr = correlation[np.argmax(np.abs(correlation))]

            st.metric(f"最大相關性時的延遲 (小時)", f"{best_lag}", help=f"當 {station1} 的時間序列移動 {best_lag} 小時後，與 {station2} 的相關係數達到最大值 {best_corr:.3f}。正值表示 {station1} 領先，負值表示落後。")
            
            fig_lag = px.line(x=lags, y=correlation, title=f"時間延遲相關性", labels={'x': '時間延遲 (小時)', 'y': '正規化相關係數'})
            fig_lag.add_vline(x=best_lag, line_width=2, line_dash="dash", line_color="red")
            st.plotly_chart(fig_lag, use_container_width=True)
        else:
            st.warning("數據點不足，無法進行有意義的時間延遲分析。")

    with adv_col2:
        st.subheader("Q-Q 分位圖 (Quantile-Quantile)")
        st.caption("比較兩組數據的分佈形狀。若數據點緊密貼合紅色對角線，表示兩者分佈非常相似。")
        quantiles = np.linspace(0.01, 0.99, 100)
        x_quantiles = mstats.mquantiles(x_series, prob=quantiles)
        y_quantiles = mstats.mquantiles(y_series, prob=quantiles)
        
        fig_qq = go.Figure()
        fig_qq.add_trace(go.Scatter(x=x_quantiles, y=y_quantiles, mode='markers', name='Quantiles'))
        fig_qq.add_trace(go.Scatter(x=[min(x_quantiles), max(x_quantiles)], y=[min(y_quantiles), max(y_quantiles)], mode='lines', name='Fit Line', line=dict(color='red', dash='dash')))
        fig_qq.update_layout(title="Q-Q 分位圖", xaxis_title=f"{station1} Quantiles", yaxis_title=f"{station2} Quantiles")
        st.plotly_chart(fig_qq, use_container_width=True)

def render_linear_analysis_plots(results, station1, station2, year, param_col):
    """渲染線性分析的所有圖表和結果"""
    merged = results["merged"]
    x_col, y_col = results["x_col"], results["y_col"]
    x_label = f"{station1} {PARAM_DISPLAY_NAMES.get(param_col, param_col)}{PARAM_UNITS.get(param_col, '')}"
    y_label = f"{station2} {PARAM_DISPLAY_NAMES.get(param_col, param_col)}{PARAM_UNITS.get(param_col, '')}"
    
    st.metric(f"皮爾森相關係數 (Pearson Correlation)", f"{results['corr']:.4f}")
    
    fig_scatter = px.scatter(
        merged, x=x_col, y=y_col, 
        trendline="ols", trendline_color_override="red", 
        marginal_x="histogram", marginal_y="histogram", 
        labels={"x": x_label, "y": y_label}, 
        title=f"{station1} vs {station2} - {PARAM_DISPLAY_NAMES.get(param_col, param_col)} 聯合分佈圖 ({year}年)"
    )
    fig_timeseries = px.line(merged, x='time', y=[x_col, y_col], title=f"{PARAM_DISPLAY_NAMES.get(param_col, param_col)} 時序比較圖")
    fig_density = px.density_heatmap(
        merged, x=x_col, y=y_col, 
        labels={"x": x_label, "y": y_label}, 
        title=f"{station1} vs {station2} - {PARAM_DISPLAY_NAMES.get(param_col, param_col)} 數據密度圖 ({year}年)"
    )

    tabs = st.tabs(["聯合分佈圖", "時序圖", "密度圖", "進階分析", "詳細數據", "下載專區"])
    with tabs[0]:
        st.info("此圖顯示兩測站數據的直接關係，並在圖表邊緣附加了各自數據的直方圖。")
        st.plotly_chart(fig_scatter, use_container_width=True)
        with st.container(border=True):
            st.markdown("##### 📈 回歸分析結果")
            st.latex(fr''' y = {results["slope"]:.4f}x {'+' if results["intercept"] >= 0 else ''} {results["intercept"]:.4f} \quad (R^2 = {results["r_squared"]:.4f})''')
    with tabs[1]:
        st.plotly_chart(fig_timeseries, use_container_width=True)
    with tabs[2]:
        st.plotly_chart(fig_density, use_container_width=True)
    with tabs[3]:
        render_advanced_analysis(merged, x_col, y_col, station1, station2)
    with tabs[4]:
        st.dataframe(merged, use_container_width=True)
    with tabs[5]:
        st.subheader("📥 下載分析產出")
        prefix = f"corr_linear_{station1}_{station2}_{param_col}_{year}"
        summary_content = f"分析報告\n{'='*20}\n測站 A: {station1}\n測站 B: {station2}\n年份: {year}\n參數: {PARAM_DISPLAY_NAMES.get(param_col, param_col)}\n\n皮爾森相關係數: {results['corr']:.4f}\n回歸線: y = {results['slope']:.4f}x + {results['intercept']:.4f}\nR-squared: {results['r_squared']:.4f}"
        
        dl_c1, dl_c2 = st.columns(2)
        with dl_c1:
            st.download_button("📥 下載圖表 (HTML)", fig_scatter.to_html(), f"{prefix}_chart.html", "text/html", use_container_width=True, key=f"dl_html_{prefix}")
            st.download_button("📥 下載數據 (CSV)", convert_df_to_csv(merged), f"{prefix}_data.csv", "text/csv", use_container_width=True, key=f"dl_csv_{prefix}")
        with dl_c2:
            st.download_button("📥 下載報告 (TXT)", summary_content, f"{prefix}_summary.txt", "text/plain", use_container_width=True, key=f"dl_txt_{prefix}")
            zip_buffer = create_download_package({f"{prefix}_chart.html": fig_scatter.to_html(), f"{prefix}_summary.txt": summary_content, f"{prefix}_data.csv": convert_df_to_csv(merged)})
            st.download_button("📦 一鍵打包下載 (.zip)", zip_buffer, f"{prefix}_package.zip", "application/zip", use_container_width=True, key=f"dl_zip_{prefix}")

def render_circular_analysis_plots(results, station1, station2, year, param_col):
    """渲染方向性分析的所有圖表和結果"""
    merged_uv = results["merged"]
    st.info("方向性參數是透過計算其U/V分量的相關性來進行評估。")
    c1, c2 = st.columns(2)
    c1.metric("U分量 (東西向) 相關係數", f"{results['corr_u']:.4f}")
    c2.metric("V分量 (南北向) 相關係數", f"{results['corr_v']:.4f}")
    
    p_name_disp, mag_col = PARAM_DISPLAY_NAMES.get(param_col, param_col), results["mag_col"]
    rose1_fig = create_rose_plot(results["df1_raw"], station1, p_name_disp, param_col, mag_col)
    rose2_fig = create_rose_plot(results["df2_raw"], station2, p_name_disp, param_col, mag_col)
    fig_u = px.scatter(merged_uv, x=f'u_{station1}', y=f'u_{station2}', trendline="ols", title=f"U分量相關性 (R={results['corr_u']:.3f})")
    fig_v = px.scatter(merged_uv, x=f'v_{station1}', y=f'v_{station2}', trendline="ols", title=f"V分量相關性 (R={results['corr_v']:.3f})")

    tabs = st.tabs(["方向玫瑰圖", "U/V分量圖", "詳細數據", "下載專區"])
    with tabs[0]:
        r_col1, r_col2 = st.columns(2)
        if rose1_fig: r_col1.plotly_chart(rose1_fig, use_container_width=True)
        else: r_col1.warning(f"{station1} 無法繪製玫瑰圖")
        if rose2_fig: r_col2.plotly_chart(rose2_fig, use_container_width=True)
        else: r_col2.warning(f"{station2} 無法繪製玫瑰圖")
    with tabs[1]:
        uv_col1, uv_col2 = st.columns(2)
        uv_col1.plotly_chart(fig_u, use_container_width=True)
        uv_col2.plotly_chart(fig_v, use_container_width=True)
    with tabs[2]:
        st.dataframe(results["merged"], use_container_width=True)
    with tabs[3]:
        st.subheader("📥 下載分析產出")
        prefix = f"corr_circular_{station1}_{station2}_{param_col}_{year}"
        summary_content = f"分析報告\n{'='*20}\n測站 A: {station1}\n測站 B: {station2}\n年份: {year}\n參數: {p_name_disp}\n\nU分量相關係數: {results['corr_u']:.4f}\nV分量相關係數: {results['corr_v']:.4f}"
        html_content = f"<html><head><title>{prefix}</title></head><body><h1>U-V分量圖</h1>" + fig_u.to_html(full_html=False, include_plotlyjs='cdn') + fig_v.to_html(full_html=False, include_plotlyjs=False) + "</body></html>"
        
        dl_c1, dl_c2 = st.columns(2)
        with dl_c1:
            st.download_button("📥 下載圖表 (HTML)", html_content, f"{prefix}_charts.html", "text/html", use_container_width=True, key=f"dl_html_{prefix}")
            st.download_button("📥 下載數據 (CSV)", convert_df_to_csv(results["merged"]), f"{prefix}_data.csv", "text/csv", use_container_width=True, key=f"dl_csv_{prefix}")
        with dl_c2:
            st.download_button("📥 下載報告 (TXT)", summary_content, f"{prefix}_summary.txt", "text/plain", use_container_width=True, key=f"dl_txt_{prefix}")
            zip_files = {
                f"{prefix}_uv_charts.html": html_content, 
                f"{prefix}_summary.txt": summary_content, 
                f"{prefix}_data.csv": convert_df_to_csv(results["merged"])
            }
            if rose1_fig: zip_files[f"{prefix}_{station1}_rose.html"] = rose1_fig.to_html()
            if rose2_fig: zip_files[f"{prefix}_{station2}_rose.html"] = rose2_fig.to_html()
            zip_buffer = create_download_package(zip_files)
            st.download_button("📦 一鍵打包下載 (.zip)", zip_buffer, f"{prefix}_package.zip", "application/zip", use_container_width=True, key=f"dl_zip_{prefix}")

def render_trend_chart_and_downloads(results_df, s1, s2, param_disp, param_col, start_y, end_y, chart_type):
    """渲染趨勢圖並提供標準化的下載選項"""
    st.subheader("📈 趨勢圖與下載")
    title = f"{s1} vs {s2} - {param_disp} 逐年相關係數 ({start_y}-{end_y})"
    plot_df = results_df.dropna(subset=['相關係數'])
    
    if plot_df.empty:
        st.warning("無足夠數據繪製趨勢圖。")
        return

    if chart_type == '長條圖':
        fig = px.bar(plot_df, x='年份', y='相關係數', text_auto='.3f', title=title)
    elif chart_type == '面積圖':
        fig = px.area(plot_df, x='年份', y='相關係數', markers=True, title=title)
    elif chart_type == '散佈圖 (含趨勢線)':
        plot_df['年份_num'] = pd.to_numeric(plot_df['年份'])
        fig = px.scatter(plot_df, x='年份_num', y='相關係數', trendline="ols", title=title, labels={"年份_num": "年份"})
    else:
        fig = px.line(plot_df, x='年份', y='相關係數', markers=True, text=plot_df['相關係數'].apply(lambda x: f'{x:.3f}'), title=title)
    
    fig.update_xaxes(type='category')
    st.plotly_chart(fig, use_container_width=True)

    if chart_type == '散佈圖 (含趨勢線)' and len(plot_df) >= 2:
        slope, intercept, r_value, _, _ = linregress(plot_df['年份_num'], plot_df['相關係數'])
        with st.container(border=True):
            st.markdown("##### 📈 趨勢線分析")
            st.latex(fr''' y = {slope:.4f}x {'+' if intercept >= 0 else ''} {intercept:.4f} \quad (R^2 = {r_value**2:.4f})''')
    
    st.markdown("---")
    st.subheader("📥 下載分析產出")
    prefix = f"trend_{s1}_{s2}_{param_col}_{start_y}-{end_y}"
    summary_content = f"分析報告\n{'='*20}\n測站 A: {s1}\n測站 B: {s2}\n年份: {start_y}-{end_y}\n參數: {param_disp}\n\n平均相關係數: {plot_df['相關係數'].mean():.4f}"
    
    dl_c1, dl_c2 = st.columns(2)
    with dl_c1:
        st.download_button("📥 下載圖表 (HTML)", fig.to_html(), f"{prefix}_chart.html", "text/html", use_container_width=True, key=f"dl_html_{prefix}")
        st.download_button("📥 下載數據 (CSV)", convert_df_to_csv(results_df), f"{prefix}_data.csv", "text/csv", use_container_width=True, key=f"dl_csv_{prefix}")
    with dl_c2:
        st.download_button("📥 下載報告 (TXT)", summary_content, f"{prefix}_summary.txt", "text/plain", use_container_width=True, key=f"dl_txt_{prefix}")
        zip_buffer = create_download_package({f"{prefix}_chart.html": fig.to_html(), f"{prefix}_summary.txt": summary_content, f"{prefix}_data.csv": convert_df_to_csv(results_df)})
        st.download_button("📦 一鍵打包下載 (.zip)", zip_buffer, f"{prefix}_package.zip", "application/zip", use_container_width=True, key=f"dl_zip_{prefix}")


# --- 主功能函式 ---
def run_single_year_analysis(locations, available_years, base_data_path):
    """執行單年度詳細比較的 UI 與邏輯"""
    st.header("單年度詳細比較")
    st.write("比較兩個測站在特定時間範圍內的數據相關性，並檢視其數據品質。")
    
    analysis_params = {}
    can_analyze = False

    with st.container(border=True):
        st.subheader("⚙️ 分析設定")
        col1, col2, col3 = st.columns(3)
        station1 = col1.selectbox('選擇測站 A:', options=locations, key='s1_single', format_func=get_station_name_from_id)
        station2 = col2.selectbox('選擇測站 B:', options=locations, key='s2_single', index=min(1, len(locations)-1), format_func=get_station_name_from_id)
        station1_name, station2_name = get_station_name_from_id(station1), get_station_name_from_id(station2)

        # <<< 修改重點 1: 即時檢查測站選擇 >>>
        # 在選擇測站後立即檢查是否相同，並提供即時反饋，而非等到按下按鈕後。
        if station1 == station2:
            col3.selectbox('選擇年份:', ["請先選擇不同測站"], disabled=True, key='y_single_disabled_same')
            st.warning("請選擇兩個不同的測站以進行比較。")
            st.selectbox('選擇參數:', ["---"], disabled=True, key='p_single_disabled_same')
            can_analyze = False
        else:
            with st.spinner(f"正在查詢 {station1_name} 與 {station2_name} 的共同可用年份..."):
                common_years = get_common_available_years(base_data_path, station1, station2, available_years)

            if not common_years:
                col3.selectbox('選擇年份:', ["無共同年份資料"], disabled=True, key='y_single_disabled_no_data')
                st.warning(f"⚠️ **{station1_name}** 與 **{station2_name}** 沒有共同的資料年份，請重新選擇測站。")
                st.selectbox('選擇參數:', ["---"], disabled=True, key='p_single_disabled_no_data')
                can_analyze = False
            else:
                # <<< 修改重點 2: 只有在條件滿足時才顯示可用選項 >>>
                # 將年份和參數選擇放在 "else" 區塊內，確保只有在找到共同年份時才讓使用者操作。
                year = col3.selectbox('選擇年份:', options=common_years, index=0, key='y_single_dynamic')
                param_options_single = {
                    "示性波高": ("Wave_Height_Significant", "linear"), "平均波週期": ("Wave_Mean_Period", "linear"),
                    "波浪尖峰週期": ("Wave_Peak_Period", "linear"), "風速": ("Wind_Speed", "linear"),
                    "陣風風速": ("Wind_Gust_Speed", "linear"), "氣溫": ("Air_Temperature", "linear"),
                    "海面溫度": ("Sea_Temperature", "linear"), "氣壓": ("Air_Pressure", "linear"),
                    "---": (None, None), "風向": ("Wind_Direction", "circular"), "波向": ("Wave_Main_Direction", "circular"),
                }
                selected_param_display = st.selectbox('選擇參數:', options=param_options_single.keys(), key='p_single')
                param_col, analysis_type = param_options_single[selected_param_display]
                
                # 只有在所有條件都滿足時，才設定分析參數並啟用按鈕
                if param_col:
                    analysis_params = {
                        "station1": station1_name, "station2": station2_name, "year": year, 
                        "param_col": param_col, "analysis_type": analysis_type
                    }
                    can_analyze = True

    # 根據 can_analyze 的狀態決定按鈕是否可被點擊
    if st.button("📊 計算單年度相關性", key='btn_single', use_container_width=True, disabled=not can_analyze):
        p = analysis_params
        # <<< 修改重點 3: 移除多餘的檢查 >>>
        # 因為上面的UI邏輯已經確保了 `can_analyze` 為 True 時，測站不同且參數有效，
        # 所以這裡不再需要 `if p.get("station1") == p.get("station2")` 的檢查。

        with st.spinner(f'正在載入與分析 {p["station1"]} vs {p["station2"]} 在 {p["year"]}年 的資料...'):
            results = calculate_single_year_correlation(base_data_path, p["station1"], p["station2"], p["year"], p["param_col"], p["analysis_type"])

        with st.expander("📊 點此查看輸入數據的品質概覽", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"##### {p['station1']} ({p['year']}年) 數據品質")
                if results["quality1"] is not None and not results["quality1"].empty:
                    st.dataframe(
                        results["quality1"].style
                            .background_gradient(cmap='Greens', subset=['完整度 (%)'])
                            .format({"完整度 (%)": "{:.2f}%", "異常值": "{:d}"}),
                        use_container_width=True
                    )
                    render_quality_pie_chart(results["quality1"], f"{p['station1']} 整體品質")
                else:
                    st.warning(f"無法載入或分析 {p['station1']} 的數據品質。")
            with col2:
                st.markdown(f"##### {p['station2']} ({p['year']}年) 數據品質")
                if results["quality2"] is not None and not results["quality2"].empty:
                    st.dataframe(
                        results["quality2"].style
                            .background_gradient(cmap='Greens', subset=['完整度 (%)'])
                            .format({"完整度 (%)": "{:.2f}%", "異常值": "{:d}"}),
                        use_container_width=True
                    )
                    render_quality_pie_chart(results["quality2"], f"{p['station2']} 整體品質")
                else:
                    st.warning(f"無法載入或分析 {p['station2']} 的數據品質。")
        st.markdown("---")
        
        if "error" in results:
            st.error(results["error"])
        else:
            st.markdown(f"#### 🔎 分析結果: **{p['station1']}** vs. **{p['station2']}** ({p['year']}年) | **{PARAM_DISPLAY_NAMES.get(p['param_col'], p['param_col'])}**")
            if results["type"] == 'linear':
                render_linear_analysis_plots(results, p["station1"], p["station2"], p["year"], p["param_col"])
            elif results["type"] == 'circular':
                render_circular_analysis_plots(results, p["station1"], p["station2"], p["year"], p["param_col"])

def run_yearly_trend_analysis(locations, available_years, base_data_path):
    """執行逐年趨勢比較的 UI 與邏輯"""
    st.header("逐年趨勢比較")
    st.write("比較兩個測站特定參數的相關性，在連續年份中的變化趨勢。")

    analysis_params = {}
    can_analyze = False

    with st.container(border=True):
        st.subheader("⚙️ 分析設定")
        col1, col2 = st.columns(2)
        s1 = col1.selectbox('測站 A:', locations, key='s1_trend', format_func=get_station_name_from_id)
        s2 = col1.selectbox('測站 B:', locations, key='s2_trend', index=min(1, len(locations)-1), format_func=get_station_name_from_id)
        s1_name, s2_name = get_station_name_from_id(s1), get_station_name_from_id(s2)
        
        param_map_trend = {
            "示性波高": "Wave_Height_Significant", "平均波週期": "Wave_Mean_Period", "波浪尖峰週期": "Wave_Peak_Period",
            "風速": "Wind_Speed", "陣風風速": "Wind_Gust_Speed", "氣溫": "Air_Temperature",
            "海面溫度": "Sea_Temperature", "氣壓": "Air_Pressure"
        }
        param_disp = col2.selectbox('參數 (僅限純量):', param_map_trend.keys(), key='p_trend')
        param_col = param_map_trend[param_disp]
        
        # <<< 修改重點 1: 同樣進行即時檢查 >>>
        if s1 == s2:
            st.warning("請選擇兩個不同的測站以進行比較。")
            st.select_slider('選擇年份範圍:', ["請先選擇不同測站"], disabled=True, key='y_slider_disabled_same')
            can_analyze = False
        else:
            with st.spinner(f"正在查詢 {s1_name} 與 {s2_name} 的共同可用年份..."):
                common_years = get_common_available_years(base_data_path, s1, s2, available_years)

            sorted_int_years = sorted([int(y) for y in common_years])

            if not sorted_int_years:
                st.select_slider('選擇年份範圍:', ["無共同年份資料"], disabled=True, key='y_slider_disabled_no_data')
                st.warning(f"⚠️ **{s1_name}** 與 **{s2_name}** 沒有共同的資料年份，請重新選擇測站。")
                can_analyze = False
            else:
                # <<< 修改重點 4: 優化年份滑桿的預設值選取邏輯 >>>
                # 確保在只有一個共同年份時不會出錯，並提供更合理的預設範圍。
                default_end = sorted_int_years[-1]
                # 預設起始值為倒數第二年，但若總年數不足則從第一年開始
                default_start_index = max(0, len(sorted_int_years) - 2)
                default_start = sorted_int_years[default_start_index]

                start_y, end_y = st.select_slider(
                    '選擇年份範圍:', 
                    options=sorted_int_years, 
                    value=(default_start, default_end), 
                    key='y_slider_dynamic'
                )
                chart_type = st.selectbox('圖表類型:', ['長條圖', '折線圖', '面積圖', '散佈圖 (含趨勢線)'], key='chart_type')
                
                analysis_params = {
                    "s1": s1_name, "s2": s2_name, "param_col": param_col, "param_disp": param_disp,
                    "start_y": start_y, "end_y": end_y, "chart_type": chart_type
                }
                can_analyze = True

    if st.button("📈 計算逐年相關性", key='btn_trend', use_container_width=True, disabled=not can_analyze):
        p = analysis_params
        # 同樣地，這裡不再需要檢查 s1 == s2

        results_df = calculate_yearly_trend(base_data_path, p["s1"], p["s2"], p["param_col"], p["start_y"], p["end_y"])
        st.success("計算完成！")
        
        if results_df['相關係數'].dropna().empty:
            st.warning("在指定的年份範圍內，沒有足夠的有效相關係數數據可供顯示。")
            st.stop()
            
        results_df['年份'] = results_df['年份'].astype(str)
        st.markdown(f"#### 🔎 分析結果: **{p['s1']}** vs. **{p['s2']}** ({p['start_y']} - {p['end_y']}年) | **{p['param_disp']}**")

        tab_chart, tab_data, tab_quality = st.tabs(["📈 趨勢圖與下載", "🔢 逐年數據", "📊 數據品質概覽"])
        
        with tab_chart:
            render_trend_chart_and_downloads(results_df, p['s1'], p['s2'], p['param_disp'], p['param_col'], p['start_y'], p['end_y'], p['chart_type'])
        with tab_data:
            st.caption("以下為逐年計算出的相關係數與用於計算的資料點數：")
            st.dataframe(results_df[['年份', '相關係數', '配對資料點數']].style.format({'相關係數': "{:.4f}"}), use_container_width=True)
        with tab_quality:
            st.info("此圖表顯示每年用於計算相關性的成對數據點數量。數量過少可能代表該年度的相關係數可信度較低。")
            fig_quality = px.bar(
                results_df, 
                x='年份', 
                y='配對資料點數',
                title='每年用於計算相關性的資料點數量',
                labels={'配對資料點數': '成對資料點數量 (筆)', '年份': '年份'},
                text_auto=True
            )
            st.plotly_chart(fig_quality, use_container_width=True)

def main():
    """主函數，根據選擇的模式調用對應的分析函式"""
    # 模擬從主頁面載入 session state
    if 'locations' not in st.session_state:
        st.session_state.locations = ['StationA', 'StationB', 'StationC']
        st.session_state.available_years = ['2021', '2022', '2023']
        st.session_state.base_data_path = '.'
        st.info("偵測到測試模式，正在使用範例資料。")

    locations = st.session_state.get('locations', [])
    base_data_path = st.session_state.get('base_data_path', '')
    available_years = st.session_state.get('available_years', [])

    if not all([locations, base_data_path, available_years]):
        st.warning("缺少必要的設定資料。請返回主頁面載入測站列表、資料路徑和可用年份。")
        return

    analysis_mode = st.radio(
        "選擇分析模組:", 
        ("單年度詳細比較", "逐年趨勢比較"), 
        horizontal=True, 
        key='main_mode'
    )
    st.markdown("---")

    if analysis_mode == "單年度詳細比較":
        run_single_year_analysis(locations, available_years, base_data_path)
    elif analysis_mode == "逐年趨勢比較":
        run_yearly_trend_analysis(locations, available_years, base_data_path)

if __name__ == "__main__":
    main()
