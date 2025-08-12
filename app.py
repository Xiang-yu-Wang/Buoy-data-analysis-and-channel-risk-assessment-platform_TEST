import streamlit as st # 確保這是第一個 Streamlit 指令的導入

# 從 helpers 匯入所有需要的函式和**已在 helpers 模組載入時初始化的全局配置變數**。
# 這樣 app.py 就不需要再次讀取 config.json 了。
from utils.helpers import initialize_session_state

# --- 頁面設定，永遠是第一個 Streamlit 指令 ---
st.set_page_config(
    page_title="浮標資料分析平台",
    page_icon="🌊",
    layout="wide"
)

initialize_session_state()

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
