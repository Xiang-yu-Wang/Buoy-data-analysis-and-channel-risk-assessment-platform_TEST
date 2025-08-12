# ==================== 完整修改版 ====================
import streamlit as st
import plotly.express as px
from utils.helpers import batch_process_all_data, convert_df_to_csv, get_station_name_from_id, initialize_session_state

import io
import zipfile

initialize_session_state()
st.title('🗺️ 航行適宜性熱力圖')
st.write('選擇年份範圍和安全閾值，分析所有測站的航行適宜性。')
st.sidebar.subheader("熱力圖設定")

# ---【關鍵修改點】---
# 已根據錯誤提示，將此變數的值修正為正確的欄位名稱。
TARGET_COLUMN_NAME = '可航行時間比例(%)' 

# 從 session_state 讀取共享資料
locations = st.session_state.get('locations', [])
base_data_path = st.session_state.get('base_data_path', '')

available_years = st.session_state.get('available_years', [])

if not available_years:
    st.warning("沒有偵測到任何可用的年份資料，請檢查資料夾設定或返回主頁面重新載入。")
    st.stop()

default_start_year = available_years[-2] if len(available_years) >= 2 else available_years[0]
default_end_year = available_years[-1] if available_years else available_years[0]

if default_start_year == default_end_year:
    st.sidebar.button(str(default_start_year), disabled=True, use_container_width=True)
    selected_start_year, selected_end_year = default_start_year, default_end_year
else:
    selected_start_year, selected_end_year = st.sidebar.select_slider(
        '選擇年份範圍:',
        options=available_years,
        value=(default_start_year, default_end_year),
        key='pages_4_hm_year_slider'
    )

wave_thresh = st.sidebar.slider("示性波高上限 (m)", 0.1, 3.0, 0.7, 0.1, key='pages_4_hm_wave_thresh')
wind_thresh = st.sidebar.slider("風速上限 (m/s)", 1.0, 20.0, 10.0, 0.5, key='pages_4_hm_wind_thresh')
view_mode = st.sidebar.radio("選擇檢視模式:", ("詳細月視圖", "年度平均視圖", "綜合季節性視圖"), key='pages_4_hm_view_mode')

if st.sidebar.button('🚀 產生熱力圖', key='pages_4_hm_button'):
    with st.spinner('正在進行批次分析...'):
        results_df, missing_sources = batch_process_all_data(
            base_data_path, 
            locations, 
            range(selected_start_year, selected_end_year + 1), 
            wave_thresh, 
            wind_thresh
        )
    
    st.success('批次分析完成！')

    if missing_sources:
        st.warning(f"注意：以下測站因找不到對應的資料檔案而未被納入分析：`{', '.join(get_station_name_from_id(loc) for loc in missing_sources)}`")

    display_df = results_df.dropna(subset=[TARGET_COLUMN_NAME])

    if not display_df.empty:
        heatmap_data = None
        if view_mode == "詳細月視圖":
            month_order = [f"{y}-{m:02d}" for y in range(selected_start_year, selected_end_year + 1) for m in range(1, 13)]
            heatmap_data = display_df.pivot(index="地點", columns="年月", values=TARGET_COLUMN_NAME)
            heatmap_data = heatmap_data.reindex(columns=[col for col in month_order if col in heatmap_data.columns])

        elif view_mode == "年度平均視圖":
            yearly_avg = display_df.groupby(['地點', '年份'])[TARGET_COLUMN_NAME].mean(numeric_only=True).reset_index()
            heatmap_data = yearly_avg.pivot(index="地點", columns="年份", values=TARGET_COLUMN_NAME)
        else: # 綜合季節性視圖
            monthly_avg = display_df.groupby(['地點', '月份'])[TARGET_COLUMN_NAME].mean(numeric_only=True).reset_index()
            month_map = {i: f"{i:02d}月" for i in range(1, 13)}
            monthly_avg['月份名稱'] = monthly_avg['月份'].map(month_map)
            heatmap_data = monthly_avg.pivot(index="地點", columns="月份名稱", values=TARGET_COLUMN_NAME)
            if not heatmap_data.empty:
                heatmap_data = heatmap_data[[month_map[i] for i in range(1, 13) if month_map[i] in heatmap_data.columns]]

        if heatmap_data is not None and not heatmap_data.empty:
            st.subheader(f'視覺化互動式熱力圖 - {view_mode}'); 
            
            heatmap_data_reindexed = heatmap_data.reindex( [get_station_name_from_id(loc) for loc in locations]).dropna(how='all')
            
            fig = px.imshow(heatmap_data_reindexed, labels=dict(x="時間", y="地點", color=TARGET_COLUMN_NAME), text_auto=".0f", aspect="auto", color_continuous_scale='Viridis_r')
            
            fig.update_layout(
                title=f"測站航行適宜性熱力圖 (波高 < {wave_thresh}m, 風速 < {wind_thresh}m/s)",
                xaxis_title="時間 / 年份 / 月份",
                yaxis_title="測站地點"
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
            st.subheader("📦 下載分析產出")

            summary_csv = convert_df_to_csv(display_df)
            pivot_csv = convert_df_to_csv(heatmap_data_reindexed)
            fig_html = fig.to_html()

            dl_col1, dl_col2, dl_col3 = st.columns(3)
            with dl_col1:
                # 修改點：將總結報告從 CSV 改為 TXT
                st.download_button("📥 下載總結報告 (TXT)", summary_csv, f"summary_{selected_start_year}-{selected_end_year}.txt", "text/plain", use_container_width=True)
            with dl_col2:
                st.download_button("📥 下載圖表數據 (CSV)", pivot_csv, f"heatmap_data_{view_mode}.csv", "text/csv", use_container_width=True)
            with dl_col3:
                st.download_button("📥 下載互動圖表 (HTML)", fig_html, f"heatmap_chart_{view_mode}.html", "text/html", use_container_width=True)

            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
                # 修改點：打包進 zip 檔的總結報告也改名為 .txt
                zip_file.writestr("summary_report.txt", summary_csv)
                zip_file.writestr("heatmap_data.csv", pivot_csv)
                zip_file.writestr("heatmap_chart.html", fig_html)
            
            st.download_button("📥 一鍵打包下載所有產出 (.zip)", zip_buffer.getvalue(), f"heatmap_package_{selected_start_year}-{selected_end_year}.zip", "application/zip", use_container_width=True)

        else:
            st.warning("在此檢視模式下無資料可顯示。")
    else:
        st.error("在指定的年份範圍內，找不到任何有效的資料來進行分析。")

# ==================== 完整修改版 (結束) ====================
