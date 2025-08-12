#最終(新增測站全選)
from altair.utils.core import P
from jinja2.utils import F
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import folium
from streamlit_folium import st_folium
from utils.helpers import initialize_session_state, load_year_data, PARAMETER_INFO, convert_df_to_csv

# --- 1. 頁面設定與標題 ---
st.set_page_config(layout="wide")
initialize_session_state()

st.title("📍 測站地圖總覽")
st.write("本地圖標示了所有已納入分析的浮標測站的地理位置，並可視覺化風場或波場動態。")
st.markdown("---")

# --- 2. 從 session_state 讀取共享資料並進行校驗 ---
devices = st.session_state['devices']
base_data_path = st.session_state['base_data_path']
available_years = st.session_state['available_years']

with st.sidebar.expander("🗺️ 測站座標診斷"):
    for device in devices:
        st.json(device, expanded=False)

# --- 3. 分析模式選擇 ---
analysis_mode = st.radio(
    "選擇地圖模式:", ("靜態地圖", "動態向量場"), key='pages_1_map_analysis_mode', horizontal=True
)

# --- 模式一：靜態地圖 ---
if analysis_mode == "靜態地圖":
    st.subheader("🌐 所有測站靜態地圖")
    
    m = folium.Map(location=[23.6, 120.6], zoom_start=8)

    for device in devices:
        folium.Marker(
            location=[device['CenterLatitude'], device['CenterLongitude']],
            popup=folium.Popup(f"<a href='/單站資料探索?station={device['StationID']}' target='_blank'>{device['Title']}</a>", max_width=300),
            icon=folium.Icon(color='blue', icon='info-sign', prefix='glyphicon')
        ).add_to(m)


    # Display map and capture interaction
    map_data = st_folium(m, width=700, height=500)


# --- 模式二：動態向量場 ---
elif analysis_mode == "動態向量場":
    st.subheader("💨🌊 動態向量場分析")
    
    with st.expander("📖 如何解讀地圖？", expanded=False):
        st.info("""
            地圖動畫將顯示向量的方向線與代表強度的箭頭。
            - **箭頭方向**：表示風或波的來向。
            - **箭頭長度與顏色**：皆代表強度（風速或波高）。長度與顏色會根據所選數據集中的最大與最小值進行縮放，並附有顏色條作為參考。
            - **箭頭尖端**：為避免動畫圖層錯誤，箭頭尖端目前將顯示為圓形，而非三角形。
            - **紅色圓點**：代表測站的實際地理位置。
        """)

    st.sidebar.header("⚙️ 向量場設定")
    selected_vector_type = st.sidebar.selectbox("選擇向量場類型:", ("風場", "波場"), key='pages_1_vector_type')

    if selected_vector_type == "風場":
        direction_col, magnitude_col, vector_title = "Wind_Direction", "Wind_Speed", "風速"
        magnitude_unit = PARAMETER_INFO.get("Wind_Speed", {}).get("unit", "m/s")
        arrow_angle_converter = lambda d: (d + 180) % 360
    else:
        direction_col, magnitude_col, vector_title = "Wave_Main_Direction", "Wave_Height_Significant", "示性波高"
        magnitude_unit = PARAMETER_INFO.get("Wave_Height_Significant", {}).get("unit", "m")
        arrow_angle_converter = lambda d: d

    select_all_stations = st.sidebar.checkbox("全選/反選所有測站", value=True, key='pages_1_select_all_stations')
    default_selection = st.session_state['devices'] if select_all_stations else []

    # use title as multiselect options
    selected_stations = st.sidebar.multiselect("選擇要顯示的測站:", options=st.session_state['devices'], default=default_selection, key='pages_1_vector_stations_select', format_func=lambda x: x['Title'])
    selected_year_for_vector = st.sidebar.selectbox("選擇年份:", options=available_years, index=len(available_years) - 1 if available_years else 0, key='pages_1_vector_year_select')
    animation_freq_options = {"每小時平均": "h", "每日平均": "D", "每週平均": "W", "每月平均": "ME"}
    selected_anim_freq_display = st.sidebar.selectbox("動畫時間間隔:", options=list(animation_freq_options.keys()), index=1, key='pages_1_anim_freq_select')
    selected_anim_freq_pandas = animation_freq_options[selected_anim_freq_display]
    
    current_params_tuple = (selected_vector_type, tuple(sorted([ device['StationID'] for device in selected_stations if 'StationID' in device ])), selected_year_for_vector, selected_anim_freq_pandas)

    if 'generated_params' in st.session_state and st.session_state.generated_params != current_params_tuple:
        st.sidebar.warning("⚠️ 設定已變更，請點擊下方按鈕重新產生圖表。")

    generate_button_pressed = st.sidebar.button("▶️ 產生向量場動畫", key='pages_1_generate_vector_button', use_container_width=True)

    if 'vector_data_cache' not in st.session_state: st.session_state.vector_data_cache = {}

    if generate_button_pressed:
        if not selected_stations:
            st.warning("請至少選擇一個測站。"); st.stop()
        
        with st.spinner(f"正在處理 {len(selected_stations)} 個測站的數據..."):
            all_vector_data_processed, skipped_stations = [], []
            progress_bar = st.progress(0, text="準備開始...")
            for i, station in enumerate(selected_stations):
                station_id = station['StationID']
                station_name = station['Title']

                progress_bar.progress((i + 1) / len(selected_stations), text=f"處理中: {station_name}")
                df_station_year = load_year_data(base_data_path, station_id, selected_year_for_vector)
                if df_station_year is None or df_station_year.empty:
                    skipped_stations.append((station_id, f"找不到 {selected_year_for_vector} 年資料")); continue
                if 'time' not in df_station_year.columns:
                    df_station_year.reset_index(inplace=True); df_station_year.rename(columns={df_station_year.columns[0]: 'time'}, inplace=True)
                df_station_year['time'] = pd.to_datetime(df_station_year['time'], errors='coerce')
                df_station_year.dropna(subset=['time', direction_col, magnitude_col], inplace=True)
                if df_station_year.empty:
                    skipped_stations.append((station_id, "必要欄位無有效數值")); continue
                df_resampled = df_station_year.set_index('time')[[direction_col, magnitude_col]].apply(pd.to_numeric, errors='coerce').resample(selected_anim_freq_pandas).mean().dropna().reset_index()
                if df_resampled.empty: continue
                current_station_coords = next((device for device in devices if device['Title'] == station_name), None)
                df_resampled['arrow_angle'] = df_resampled[direction_col].apply(arrow_angle_converter)
                df_resampled['station_name'] = station_id
                df_resampled['lat'] = current_station_coords['CenterLatitude'] if current_station_coords else np.nan
                df_resampled['lon'] = current_station_coords['CenterLongitude'] if current_station_coords else np.nan
                all_vector_data_processed.append(df_resampled)
            progress_bar.empty()

            if not all_vector_data_processed:
                st.error("無任何有效數據可供顯示。"); st.session_state.vector_data_cache = {}; st.stop()

            combined_vector_df = pd.concat(all_vector_data_processed, ignore_index=True).sort_values(by='time').dropna()
            if combined_vector_df.empty:
                st.error("最終數據為空，無法生成動畫。"); st.session_state.vector_data_cache = {}; st.stop()

            all_magnitudes = combined_vector_df[magnitude_col]
            min_mag, max_mag = all_magnitudes.min(), all_magnitudes.max()
            min_arrow, max_arrow = 0.054, 0.54
            combined_vector_df['normalized_magnitude'] = 0.5 if pd.isna(max_mag) or (max_mag - min_mag) < 1e-6 else (all_magnitudes - min_mag) / (max_mag - min_mag)
            combined_vector_df['dynamic_arrow_length'] = combined_vector_df['normalized_magnitude'] * (max_arrow - min_arrow) + min_arrow
            combined_vector_df['end_lat'] = combined_vector_df['lat'] + combined_vector_df['dynamic_arrow_length'] * np.cos(np.radians(90 - combined_vector_df['arrow_angle']))
            combined_vector_df['end_lon'] = combined_vector_df['lon'] + combined_vector_df['dynamic_arrow_length'] * np.sin(np.radians(90 - combined_vector_df['arrow_angle']))
            combined_vector_df['time_str'] = combined_vector_df['time'].dt.strftime('%Y-%m-%d %H:%M')
            st.session_state.generated_params = current_params_tuple
            st.session_state.vector_data_cache = {
                'df': combined_vector_df, 'min_magnitude': min_mag, 'max_magnitude': max_mag, 'skipped': skipped_stations,
                'params_display': {'vector_title': vector_title, 'magnitude_unit': magnitude_unit, 'selected_year': selected_year_for_vector,
                                   'selected_freq': selected_anim_freq_display, 'direction_col': direction_col, 'magnitude_col': magnitude_col}}
            st.rerun()

    if 'df' in st.session_state.vector_data_cache and not st.session_state.vector_data_cache['df'].empty:
        cached_data = st.session_state.vector_data_cache
        df = cached_data['df']
        min_mag_plot, max_mag_plot = cached_data['min_magnitude'], cached_data['max_magnitude']
        skipped = cached_data['skipped']
        params = cached_data['params_display']
        magnitude_col, direction_col = params['magnitude_col'], params['direction_col']

        tab1, tab2, tab3, tab4 = st.tabs(["🗺️ 動態地圖", "📊 數據摘要", "📥 資料下載", "⚠️ 處理日誌"])

        with tab1:
            st.markdown(f"**當前數據集強度範圍：** `{min_mag_plot:.2f}` ~ `{max_mag_plot:.2f}` {params['magnitude_unit']}")
            
            unique_times = sorted(df['time_str'].unique())
            initial_time_str = unique_times[-1]
            initial_df = df[df['time_str'] == initial_time_str]
            
            initial_lines_lat, initial_lines_lon = [], []
            for _, row in initial_df.iterrows():
                initial_lines_lat.extend([row['lat'], row['end_lat'], None])
                initial_lines_lon.extend([row['lon'], row['end_lon'], None])
            
            fig = go.Figure(data=[
                go.Scattermapbox(lat=initial_lines_lat, lon=initial_lines_lon, mode='lines', line=dict(width=2.5, color='rgba(0, 115, 230, 0.8)'), hoverinfo='none', showlegend=False),
                go.Scattermapbox(
                    lat=initial_df['end_lat'], lon=initial_df['end_lon'], mode='markers',
                    marker=dict(symbol='circle', size=12, color=initial_df[magnitude_col], colorscale='Viridis', cmin=min_mag_plot, cmax=max_mag_plot, showscale=True,
                                colorbar=dict(title=f"<b>{params['vector_title']}</b><br>({params['magnitude_unit']})", x=1.01, y=0.5, len=0.7, thickness=15, yanchor='middle', xanchor='left')),
                    hovertemplate=f"<b>{params['vector_title']}:</b> %{{marker.color:.2f}}<extra></extra>",
                    showlegend=False
                ),
                go.Scattermapbox(
                    lat=initial_df['lat'], lon=initial_df['lon'], mode='markers', marker=dict(size=8, color='red', opacity=0.7),
                    text=initial_df['station_name'], customdata=np.stack((initial_df[magnitude_col], initial_df[direction_col]), axis=-1),
                    hovertemplate='<b>%{text}</b><br>' + f"<b>{params['vector_title']}:</b> %{{customdata[0]:.2f}} {params['magnitude_unit']}<br>" + '<b>方向:</b> %{customdata[1]:.1f}°<extra></extra>',
                    showlegend=False
                )
            ])

            frames = []
            for time_str in unique_times:
                frame_df = df[df['time_str'] == time_str]
                lines_lat, lines_lon = [], []
                for _, row in frame_df.iterrows():
                    lines_lat.extend([row['lat'], row['end_lat'], None])
                    lines_lon.extend([row['lon'], row['end_lon'], None])
                
                frames.append(go.Frame(name=time_str, data=[
                    go.Scattermapbox(lat=lines_lat, lon=lines_lon),
                    go.Scattermapbox(lat=frame_df['end_lat'], lon=frame_df['end_lon'], marker={'color': frame_df[magnitude_col]}),
                    go.Scattermapbox(lat=frame_df['lat'], lon=frame_df['lon'], text=frame_df['station_name'], customdata=np.stack((frame_df[magnitude_col], frame_df[direction_col]), axis=-1))
                ], traces=[0, 1, 2]))

            fig.frames = frames
            
            # --- 修改重點：調整 mapbox 中心點、縮放等級和邊距 ---
            fig.update_layout(
                mapbox_style="open-street-map",
                # 固定中心點在台灣，並設定合適的縮放等級
                mapbox_center={"lat": 23.9, "lon": 121.0},
                mapbox_zoom=6.5,
                title_text=f"動態向量場: {params['vector_title']} ({params['selected_year']}年, {params['selected_freq']})", title_x=0.5,
                # 稍微減少底部邊距，讓地圖看起來更大
                margin={"r":0,"t":50,"l":0,"b":70},
                
                updatemenus=[dict(
                    type="buttons",
                    showactive=True,
                    y=-0.1, x=0.1, yanchor="top", xanchor="right",
                    font=dict(color='black', size=12),
                    buttons=[dict(
                        label="▶️ 播放",
                        method="animate",
                        args=[None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}, "mode": "immediate"}]
                    )]
                )],
                
                sliders=[dict(
                    active=len(unique_times)-1,
                    y=-0.1, x=0.55, len=0.8, yanchor="top", xanchor="center",
                    currentvalue={"font": {"size": 12}, "prefix": "時間: ", "visible": True, "xanchor": "right"},
                    transition={"duration": 0},
                    steps=[dict(
                        method="animate",
                        args=[[f.name], {"frame": {"duration": 500, "redraw": True}, "mode": "immediate"}],
                        label=f.name
                    ) for f in frames]
                )]
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader("📊 數據品質與統計概覽")
            start_date, end_date = df['time'].min(), df['time'].max()
            col1, col2, col3 = st.columns(3)
            col1.metric("數據起點", start_date.strftime('%Y-%m-%d')); col2.metric("數據終點", end_date.strftime('%Y-%m-%d')); col3.metric("總筆數 (已重採樣)", f"{len(df):,}")
            st.markdown("##### **數據品質**")
            st.write(f"**分析參數:** {params['vector_title']}")
            data_series = df[magnitude_col]
            q_col1, q_col2, q_col3 = st.columns(3)
            q_col1.metric("平均值", f"{data_series.mean():.2f} {params['magnitude_unit']}"); q_col2.metric("最大值", f"{data_series.max():.2f} {params['magnitude_unit']}"); q_col3.metric("最小值", f"{data_series.min():.2f} {params['magnitude_unit']}")
            st.dataframe(data_series.describe().to_frame().T.round(2), use_container_width=True)

        with tab3:
            st.subheader("📦 下載處理後的數據與圖表")
            st.info("點擊下方按鈕，可將目前圖表中使用的數據或互動式圖表檔案下載到你的電腦中。")
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(label="📥 下載數據 (CSV)", data=convert_df_to_csv(df), file_name=f"vector_data_{selected_vector_type}_{params['selected_year']}.csv", mime="text/csv", use_container_width=True)
            with col2:
                download_fig = fig.to_html(full_html=False, include_plotlyjs='cdn').encode('utf-8')
                st.download_button(label="📥 下載圖表 (HTML)", data=download_fig, file_name=f"vector_chart_{selected_vector_type}_{params['selected_year']}.html", mime="text/html", use_container_width=True)

        with tab4:
            st.subheader("⚠️ 處理日誌")
            if skipped:
                st.warning("部分測站因資料問題在處理過程中被跳過：")
                for name, reason in skipped: st.markdown(f"- **{name}**: {reason}")
            else:
                st.success("所有選擇的測站均已成功處理。")
    else:
        st.info("⬅️ 請在左側側邊欄中設定參數，然後點擊 **「產生向量場動畫」** 來載入並顯示地圖。")
