# 🌊 浮標資料分析與近岸作業適宜性評估平台

## 專案簡介

本專案是一個基於 Streamlit 開發的互動式網頁應用程式，專為海洋浮標觀測資料的分析與視覺化而設計。它不僅提供多站點的資料探索、統計分析與相關性研究，更整合了深度學習模型進行時間序列預測，最終目標是**評估特定海域的航行適宜性**，為航運、漁業及海洋研究提供數據驅動的決策支援。

## 核心功能

*   **📈 全方位視覺化探索**: 提供測站地理分佈、單站數據趨勢、統計儀表板及風玫瑰圖，實現從宏觀到微觀的資料洞察。
*   **🚢 航行適宜性評估**: 透過熱力圖直觀展示不同時間點的航行風險，並結合多維度數據進行綜合評估。
*   **🧠 智慧化海象預測**: 整合 LSTM、GRU 等先進的深度學習模型，對關鍵海洋參數（如波高、風速）進行未來趨勢預測。
*   **🔗 測站關聯性分析**: 計算並視覺化不同測站之間的數據相關性，協助發現區域性的海洋現象。
*   **🔧 模組化與高擴展性**: 應用程式的每個核心功能都被設計成獨立頁面，便於未來的功能擴充與維護。

## 🛠️ 技術棧 (Technology Stack)

- **前端框架**: [Streamlit](https://streamlit.io/)
- **數據處理**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **資料視覺化**: [Plotly](https://plotly.com/python/), [Matplotlib](https://matplotlib.org/), [Windrose](https://github.com/python-windrose/windrose)
- **機器學習/深度學習**: [TensorFlow](https://www.tensorflow.org/), [Scikit-learn](https://scikit-learn.org/), [Prophet](https://facebook.github.io/prophet/)
- **時間序列分析**: [ruptures](https://centre-borelli.github.io/ruptures-docs/), [pmdarima](https://alkaline-ml.com/pmdarima/)

## 🎯 使用對象

本平台主要服務於以下領域的專業人士與研究者：

*   **航運業者與港務管理人員**: 用於評估航線風險，優化航行計畫。
*   **海洋科學研究者**: 用於探索海象數據、驗證海洋模型。
*   **漁業從業者**: 了解作業海域的環境變化，確保出航安全。
*   **數據分析師與工程師**: 作為一個結合 Streamlit 與時間序列分析的應用範例。

## 🚀 如何開始 (Getting Started)

請依照以下步驟在本機端運行此專案：

1.  **前置需求**
    - 確認已安裝 Python 3.9 ~ Python 3.12。

2.  **建立並啟動虛擬環境**
    - 確認您的作業系統支援 `virtualenv`。
    - 在專案根目錄執行以下指令：

    首先，建立一個虛擬環境來隔離專案的依賴套件：
    ```bash
    python -m venv .venv
    ```

    接著，根據您的作業系統啟動虛擬環境：

    - **Windows (使用 PowerShell):**
      ```powershell
      .venv\Scripts\Activate.ps1
      ```

    - **Windows (使用 Command Prompt):
      ```batch
      .venv\Scripts\activate.bat
      ```

    - **macOS / Linux (使用 bash/zsh):**
      ```bash
      source .venv/bin/activate
      ```

3.  **安裝相依套件**
    ```bash
    pip install -r requirements.txt
    ```

4.  **執行應用程式**
    ```bash
    streamlit run app.py
    ```

5.  開啟您的瀏覽器並前往 `http://localhost:8501`。

## 主要架構概覽

本專案採用模組化的結構設計，將應用程式拆分為設定、主程式、功能頁面與共用工具，以實現高內聚、低耦合的目標。

```
浮標資料航道分析必要/
├── .venv/                  # 虛擬環境目錄
├── Noto_Sans_TC/           # 專案使用的字體檔案
├── pages/                  # Streamlit 功能頁面模組
│   ├── 1_📍_測站地圖總覽.py
│   ├── 2_🔬_單站資料探索.py
│   └── ...                 # 其他獨立的分析頁面
├── utils/                  # 共用工具模組
│   └── helpers.py          # 輔助函式 (資料讀取、圖表設定等)
├── 資料檔/                 # 專案所需資料
│   ├── 浮標資料/           # 各測站的觀測數據 (CSV)
│   └── 座標CSV/              # 測站地理座標資訊
├── app.py                  # 應用程式主入口 (負責全局設定與導航)
├── config.json             # 專案的核心設定檔
├── requirements.txt        # Python 相依套件列表
└── README.md               # 本文件
```

### 關鍵模組說明

-   `app.py`: 作為應用程式的**主入口**，負責載入全局設定、初始化 Streamlit 環境，並定義側邊欄的導航結構。
-   `pages/`: 此目錄下的每個 `.py` 檔案都對應一個獨立的**功能頁面**。Streamlit 會自動將這些檔案渲染為應用程式的不同分頁，例如「測站地圖總覽」、「單站資料探索」等。
-   `utils/helpers.py`: 一個**共用工具模組**，封裝了重複使用的函式，如資料載入、數據清理、圖表樣式設定等，確保程式碼的 DRY (Don't Repeat Yourself) 原則。
-   `config.json`: **集中式設定檔**，用於管理專案中的所有可變參數，例如檔案路徑、站點座標、參數單位等。這種設計讓非程式碼的修改變得更加方便，無需更動主要邏輯。

## 📊 功能展示

-   **測站地圖總覽**: 在互動式地圖上標示所有浮標測站的位置，點擊可查看基本資訊。
-   **單站資料探索**: 選擇特定測站與時間範圍，視覺化顯示波高、風速等多種參數的時間序列圖表。
-   **航行適宜性熱力圖**: 根據使用者定義的風險閾值（如示性波高、最大波高），生成視覺化的熱力圖，快速識別高風險時段。
-   **時間序列預測**: 利用 LSTM 或 GRU 模型，對未來數小時或數天的海象數據進行預測，並將預測結果與歷史數據進行對比。

## 📝 未來展望

-   **模型優化**: 引入更多先進的時間序列預測模型，並提供超參數調整介面。
-   **即時資料整合**: 嘗試接入即時浮標資料 API，實現近乎即時的近岸作業風險評估。
-   **使用者自訂分析**: 開放讓使用者上傳自己的數據集進行分析。
-   **警報系統**: 當預測數據超過安全閾值時，自動發送通知。

## 🤝 貢獻 (Contributing)

我們非常歡迎任何形式的貢獻！如果您有任何新功能的建議、發現程式中的錯誤，或是希望優化現有流程，請隨時提出 Issue 或提交 Pull Request。

若有任何想法，也歡迎直接聯繫：`cdmnnia4131@gmail.com`

## 📄 授權 (License)

本專案採用 MIT License 授權。詳情請參閱 `LICENSE` 檔案。
