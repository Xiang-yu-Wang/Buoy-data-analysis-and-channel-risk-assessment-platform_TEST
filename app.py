import streamlit as st # 確保這是第一個 Streamlit 指令的導入
import os
import pandas as pd
import json

# 從 helpers 匯入所有需要的函式和**已在 helpers 模組載入時初始化的全局配置變數**。
# 這樣 app.py 就不需要再次讀取 config.json 了。
from utils.helpers import (
    get_available_years,
    set_chinese_font_for_matplotlib, # 使用新的 Matplotlib 字體設定函數名稱
    load_app_config_and_font, # 雖然主要目的是初始化 helpers 內部全局變數，在這裡呼叫一次可確保 Streamlit 環境下的配置載入流程
    PARAMETER_INFO, # 直接匯入 helpers 中已初始化的全局變數
    DATA_SUBFOLDERS_PRIORITY, # 直接匯入 helpers 中已初始化的全局變數
    BASE_DATA_PATH_FROM_CONFIG, # 直接匯入 helpers 中已初始化的全局變數 (注意名稱已變更)
    CHINESE_FONT_NAME, # 直接匯入 helpers 中已初始化的全局變數 (用於 Plotly)
    CHINESE_FONT_PATH_FULL, # 直接匯入 helpers 中已初始化的全局變數 (用於 Matplotlib)
    STATION_COORDS # 直接匯入 helpers 中已初始化的全局變數
)

# --- 頁面設定，永遠是第一個 Streamlit 指令 ---
st.set_page_config(
    page_title="浮標資料分析平台",
    page_icon="🌊",
    layout="wide"
)

# --- 在 set_page_config 之後安全地確認配置已載入 ---
# 這裡再次呼叫 load_app_config_and_font()，主要是為了在 Streamlit UI 上顯示載入配置時可能發生的錯誤。
# 實際的全局變數初始化已在 helpers.py 模組載入時完成。
try:
    load_app_config_and_font()
except FileNotFoundError:
    st.error(f"錯誤：配置檔 'config.json' 不存在。請確保它與 app.py 在同一個資料夾。")
    st.stop()
except json.JSONDecodeError as e:
    st.error(f"錯誤：配置檔 'config.json' 格式不正確。請檢查 JSON 語法。錯誤訊息: {e}")
    st.stop()
except Exception as e:
    st.error(f"載入配置檔時發生未知錯誤: {e}")
    st.stop()

# --- 初始化 Session State，讓所有頁面能共享資料 ---
def initialize_session_state():
    # 直接使用從 helpers 匯入的全局變數來初始化 Session State
    if 'base_data_path' not in st.session_state:
        st.session_state.base_data_path = BASE_DATA_PATH_FROM_CONFIG
    if 'chinese_font_path' not in st.session_state:
        st.session_state.chinese_font_path = CHINESE_FONT_PATH_FULL
    if 'chinese_font_name' not in st.session_state: # 新增：Plotly 需要字體名稱
        st.session_state.chinese_font_name = CHINESE_FONT_NAME
    if 'station_coords' not in st.session_state:
        st.session_state.station_coords = STATION_COORDS
    if 'parameter_info' not in st.session_state:
        st.session_state.parameter_info = PARAMETER_INFO
    if 'data_subfolders_priority' not in st.session_state:
        st.session_state.data_subfolders_priority = DATA_SUBFOLDERS_PRIORITY

    if 'locations' not in st.session_state:
        # 建構完整的資料路徑 (考慮到 config.json 中的可能是相對路徑)
        full_base_data_path_abs = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), st.session_state.base_data_path))
        try:
            if not os.path.exists(full_base_data_path_abs):
                st.session_state.locations = []
                st.error(f"錯誤：配置檔中指定的資料路徑 **'{full_base_data_path_abs}'** 不存在。請檢查配置檔 **'config.json'** 中的 **'base_data_path'** 設定。")
                return # 提前返回，避免後續操作出錯

            st.session_state.locations = sorted([item for item in os.listdir(full_base_data_path_abs) if os.path.isdir(os.path.join(full_base_data_path_abs, item)) and not item.startswith('.')])
        except Exception as e:
            st.session_state.locations = []
            st.error(f"載入測站列表時發生錯誤: {e}")

    if 'available_years' not in st.session_state:
        if st.session_state.locations and st.session_state.base_data_path:
            # get_available_years 會自行處理相對路徑轉換為絕對路徑
            st.session_state.available_years = get_available_years(st.session_state.base_data_path, st.session_state.locations)
        else:
            st.session_state.available_years = []

# --- 主程式執行區 ---
initialize_session_state()

# 應用 Matplotlib 中文字體設定 (如果需要 Matplotlib 圖表)
# 如果 main app.py 頁面不直接繪製 Matplotlib 圖，這行程式碼也可以移動到需要繪圖的子頁面中。
if st.session_state.chinese_font_path and st.session_state.chinese_font_name:
    set_chinese_font_for_matplotlib(st.session_state.chinese_font_path, st.session_state.chinese_font_name)


st.title("🌊 浮標資料分析平台")
st.markdown("---")
st.header("歡迎使用！")
st.write("請從左側的側邊欄選擇您想使用的分析功能。")

# 顯示錯誤或提示
if not st.session_state.locations:
    st.error(f"錯誤：在主資料夾 **'{st.session_state.base_data_path}'** 下找不到任何測站子資料夾，或主資料夾不存在。請檢查配置檔 **'config.json'** 中的 **'base_data_path'** 設定。")
    st.stop()

st.info(f"成功偵測到 **{len(st.session_state.locations)}** 個測站資料夾。")
st.sidebar.success("請從上方選擇一個頁面開始分析。")

##建立虛擬環境: python3 -m venv|||.venv python -m venv .venv
##啟動虛擬環境: source .venv/bin/activate(Mac)|||.venv\Scripts\Activate.ps1(Windows)
##安裝所有套件: pip install -r requirements.txt
##執行 App: streamlit run app.py\
##網址： http://localhost:8501 
