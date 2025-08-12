from glob import glob
import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import re
import chardet

# Optional: For Matplotlib related functions, if you still use them in other parts of your app
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# --- 全局配置變數 (在模組載入時初始化) ---
PARAMETER_INFO = {}
RISK_THRESHOLDS = {}
DATA_SUBFOLDERS_PRIORITY = []
BASE_DATA_PATH_FROM_CONFIG = "/dataset/buoy"
CHINESE_FONT_NAME = None # 用於 Streamlit Plotly 圖表的中文字體名稱
CHINESE_FONT_PATH_FULL = None # 用於 Matplotlib 的中文字體完整路徑

# --- 載入配置檔 ---
def load_app_config_and_font():
    """載入應用程式的配置檔，並設定中文字體資訊。
    此函數會更新模組層級的全局變數。
    """
    global PARAMETER_INFO, DATA_SUBFOLDERS_PRIORITY, BASE_DATA_PATH_FROM_CONFIG, RISK_THRESHOLDS
    global CHINESE_FONT_NAME, CHINESE_FONT_PATH_FULL, STATION_COORDS 

    CONFIG_FILE_NAME = 'config.json'
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.normpath(os.path.join(current_dir, '..', CONFIG_FILE_NAME))
    
    try:
        if not os.path.exists(config_path):
            print(f"錯誤: 配置檔 '{config_path}' 不存在。請確保它與 Streamlit 應用程式的入口檔案在同一個資料夾。")
            raise FileNotFoundError(f"配置檔 '{config_path}' 不存在。")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        if "STATION_COORDS" in config:
            new_coords = {}
            for station, coords in config["STATION_COORDS"].items():
                new_coords[station] = {
                    "latitude": coords.get("lat", coords.get("latitude")),
                    "longitude": coords.get("lon", coords.get("longitude"))
                }
            STATION_COORDS = new_coords
        
        PARAMETER_INFO = config.get("PARAMETER_INFO", {})
        DATA_SUBFOLDERS_PRIORITY = config.get("DATA_SUBFOLDERS_PRIORITY", ["qc", "QC", "real time", "real_time", "RealTime", "Real Time", "realtime"])
        BASE_DATA_PATH_FROM_CONFIG = config.get("base_data_path", "/dataset/buoy")
        RISK_THRESHOLDS = config.get("RISK_THRESHOLDS", {})

        font_path_relative = config.get("CHINESE_FONT_PATH")
        if font_path_relative:
            full_font_path = os.path.normpath(os.path.join(current_dir, '..', font_path_relative))
            if os.path.exists(full_font_path):
                CHINESE_FONT_NAME = os.path.splitext(os.path.basename(full_font_path))[0]
                CHINESE_FONT_PATH_FULL = full_font_path
                # print(f"中文字型 '{CHINESE_FONT_NAME}' 成功載入。")
            else:
                print(f"警告: 配置檔中指定的中文字型檔 '{full_font_path}' 不存在。將使用預設字型。")
        else:
            print("警告: 配置檔中未指定中文字型路徑。將使用預設字型。")

        return config

    except Exception as e:
        print(f"載入配置檔時發生錯誤: {e}")
        raise

try:
    load_app_config_and_font()
except Exception as e:
    print(f"應用程式啟動時配置初始化失敗: {e}. 請檢查您的 'config.json' 檔案。")
    PARAMETER_INFO, DATA_SUBFOLDERS_PRIORITY, STATION_COORDS = {}, [], {}
    CHINESE_FONT_NAME, CHINESE_FONT_PATH_FULL = None, None

# --- 輔助函數 ---

@st.cache_resource
def set_chinese_font_for_matplotlib(font_path_full, font_name):
    """設定 Matplotlib 的中文字型。"""
    try:
        if os.path.exists(font_path_full):
            my_font_prop = fm.FontProperties(fname=font_path_full)
            plt.rcParams['font.sans-serif'] = [my_font_prop.get_name()]
            plt.rcParams['axes.unicode_minus'] = False
        else:
            print(f"警告：中文字型檔 '{font_path_full}' 不存在。")
    except Exception as e:
        print(f"設定 Matplotlib 字型時發生錯誤: {e}")
        plt.rcParams['font.sans-serif'] = ['sans-serif']
        plt.rcParams['axes.unicode_minus'] = False

def convert_df_to_csv(df):
    """將 DataFrame 轉換為可供下載的 CSV 格式。"""
    if df is None or df.empty:
        return pd.DataFrame().to_csv(index=False).encode('utf-8')
    return df.to_csv(index=False).encode('utf-8')

@st.cache_data(ttl=3600)
def load_single_file(file_path):
    """載入並清理單一月份的 CSV 檔案。"""
    df = None
    try:
        # 自動偵測編碼
        with open(file_path, 'rb') as f:
            raw_data = f.read(100000)
            result = chardet.detect(raw_data)
            detected_encoding = result['encoding']

        encodings_to_try = list(dict.fromkeys([detected_encoding, 'utf-8', 'big5', 'cp950', 'gbk', 'latin-1']))

        for encoding in encodings_to_try:
            if encoding:
                try:
                    df = pd.read_csv(file_path, delimiter=',', on_bad_lines='warn', header=None, low_memory=False, encoding=encoding)
                    break
                except (UnicodeDecodeError, pd.errors.ParserError):
                    continue

        if df is None or len(df) < 3:
            return None

        # 處理標頭並清理欄位名稱
        english_headers = df.iloc[1].astype(str).str.strip()
        df.columns = english_headers
        df = df[3:].reset_index(drop=True)
        df.columns = df.columns.str.strip()

        # --- 自動尋找並重新命名時間欄位 ---
        time_col_candidates = ['time', 'Time', '觀測時間', 'Date', 'datetime', 'Datetime', '時間']
        actual_time_col = None
        for col in time_col_candidates:
            if col in df.columns:
                actual_time_col = col
                break
        
        if actual_time_col and actual_time_col != 'time':
            df.rename(columns={actual_time_col: 'time'}, inplace=True)
        
        # 轉換數值型別
        cols_to_convert = [col for col, info in PARAMETER_INFO.items() if info.get('type') in ['linear', 'circular']]
        for col in cols_to_convert:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 轉換時間型別
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'], errors='coerce')
            df.dropna(subset=['time'], inplace=True)
            if df['time'].empty:
                return None
        
        return df
    except Exception:
        return None

@st.cache_data(ttl=3600)
def load_year_data(base_data_path, station, year):
    """載入並合併指定測站和年份的所有月份資料。"""
    monthly_dfs = []
    dataset_path = os.path.join(base_data_path, station)

    # 載入該年度所有月份的檔案
    for month in range(1, 13):
        file_path = os.path.join(dataset_path, f"{year}{month:02d}.csv")
        if os.path.exists(file_path):
            df_month = load_single_file(file_path)
            if df_month is not None and not df_month.empty:
                monthly_dfs.append(df_month)
                
    if not monthly_dfs:
        return None

    combined_df = pd.concat(monthly_dfs, ignore_index=True)
    
    if 'time' in combined_df.columns:
        if combined_df['time'].isnull().all():
            return None # 如果所有時間值都無效，返回 None
        
        combined_df = combined_df.sort_values(by='time').drop_duplicates(subset=['time'], keep='first')
    else:
        return None

    return combined_df.reset_index(drop=True)

def load_data_for_prediction_page(station_name, param_col, start_date, end_date):
    df_list = []
    base_data_path_full = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', BASE_DATA_PATH_FROM_CONFIG))
    years_to_load = range(start_date.year, end_date.year + 1)
    
    for year in years_to_load:
        df_year = load_year_data(base_data_path_full, station_name, year)
        if df_year is not None and not df_year.empty and 'time' in df_year.columns:
            df_year = df_year.set_index('time')
            df_list.append(df_year)
    
    if not df_list: return pd.DataFrame()

    combined_df = pd.concat(df_list)
    combined_df = combined_df[~combined_df.index.duplicated(keep='first')].sort_index()

    combined_df = combined_df[
        (combined_df.index.to_series() >= pd.to_datetime(start_date)) &
        (combined_df.index.to_series() <= pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1))
    ].copy()

    if combined_df.empty or param_col not in combined_df.columns: return pd.DataFrame()
    
    df_filtered = combined_df[[param_col]].copy()
    df_filtered.columns = ['y']
    df_filtered.reset_index(inplace=True)
    df_filtered.rename(columns={'time': 'ds'}, inplace=True)

    return df_filtered

def create_sequences(data, look_back):
    X, y = [], []
    if data.ndim == 1: data = data.reshape(-1, 1)
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), :])
        y.append(data[i + look_back, 0])
    return np.array(X), np.array(y)

def analyze_data_quality(df, relevant_params=None):
    quality_metrics = {}
    if df.empty: return quality_metrics

    if relevant_params is None:
        relevant_params = [col for col, info in PARAMETER_INFO.items() if info.get('type') == 'linear']
    
    actual_params_to_check = [param for param in relevant_params if param in df.columns]

    for param_col in actual_params_to_check:
        total_records, missing_count = len(df), df[param_col].isnull().sum()
        valid_count = total_records - missing_count
        missing_percentage = (missing_count / total_records) * 100 if total_records > 0 else 0

        if pd.api.types.is_numeric_dtype(df[param_col]):
            series_clean = df[param_col].dropna()
            Q1, Q3 = series_clean.quantile(0.25), series_clean.quantile(0.75)
            IQR = Q3 - Q1
            outlier_iqr_count = 0
            if IQR > 1e-9:
                lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
                outlier_iqr_count = series_clean[(series_clean < lower) | (series_clean > upper)].count()
            
            quality_metrics[param_col] = {
                'total_records': total_records, 'valid_count': valid_count, 'missing_count': missing_count,
                'missing_percentage': missing_percentage, 'zero_count': (df[param_col] == 0).sum(),
                'negative_count': (df[param_col] < 0).sum(), 'outlier_iqr_count': outlier_iqr_count,
                'is_numeric': True
            }
        else:
            quality_metrics[param_col] = {'is_numeric': False}
    return quality_metrics

def prepare_windrose_data(df):
    if df is None or df.empty or 'Wind_Speed' not in df.columns or 'Wind_Direction' not in df.columns: return None
    df_wind = df[['Wind_Speed', 'Wind_Direction']].copy()
    df_wind['Wind_Speed'] = pd.to_numeric(df_wind['Wind_Speed'], errors='coerce')
    df_wind['Wind_Direction'] = pd.to_numeric(df_wind['Wind_Direction'], errors='coerce')
    df_wind.dropna(inplace=True)
    if df_wind.empty: return None

    dir_bins = np.arange(-11.25, 370.0, 22.5)
    dir_labels = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    speed_bins = [-1, 2, 4, 6, 8, 10, 12, np.inf]
    speed_labels = ['0-2 m/s', '2-4 m/s', '4-6 m/s', '6-8 m/s', '8-10 m/s', '10-12 m/s', '>12 m/s']

    df_wind['direction_bin'] = pd.cut((df_wind['Wind_Direction'] + 11.25) % 360, bins=dir_bins, labels=dir_labels, right=False)
    df_wind['speed_bin'] = pd.cut(df_wind['Wind_Speed'], bins=speed_bins, labels=speed_labels, right=True)

    windrose_df = df_wind.groupby(['direction_bin', 'speed_bin'], observed=False).size().reset_index(name='frequency')
    windrose_df['percentage'] = (windrose_df['frequency'] / len(df_wind)) * 100
    return windrose_df

def get_available_years(base_data_path_from_config, locations):
    all_years = set()
    base_data_path_full = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', base_data_path_from_config))
    if not os.path.exists(base_data_path_full):
        current_year = pd.Timestamp.now().year
        return list(range(current_year - 5, current_year + 1))

    for location in locations:
        location_path = os.path.join(base_data_path_full, location)
        if not os.path.isdir(location_path): continue
        
        for dirpath, _, filenames in os.walk(location_path):
            for filename in filenames:
                match = re.match(r'(\d{4})\d{2}\.csv', filename, re.IGNORECASE)
                if match:
                    all_years.add(int(match.group(1)))
                    
    if not all_years:
        current_year = pd.Timestamp.now().year
        return list(range(current_year - 5, current_year + 1))
    return sorted(list(all_years))

def batch_process_all_data(base_data_path_from_config, locations, years_to_analyze, wave_thresh, wind_thresh):
    all_results = []
    missing_data_sources = []
    base_data_path_full = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', base_data_path_from_config))

    for location in locations:
        has_data_for_location = False
        for year in years_to_analyze:
            df_year = load_year_data(base_data_path_full, location, year)
            if df_year is not None and not df_year.empty:
                has_data_for_location = True
                
                # --- 第 1 個修改：將 'month' 欄位直接命名為 '月份' ---
                df_year['月份'] = df_year['time'].dt.month
                
                # --- 第 2 個修改：使用 '月份' 進行分組 ---
                monthly_results = df_year.groupby('月份').apply(
                    lambda df_month: analyze_navigability(df_month, wave_thresh, wind_thresh)
                ).reset_index(name='可航行時間比例(%)')
                
                monthly_results['地點'] = get_station_name_from_id(location)
                monthly_results['年份'] = year
                
                # --- 現在這行可以正常運作，因為 '月份' 欄位存在 ---
                monthly_results['年月'] = monthly_results.apply(lambda row: f"{row['年份']}-{int(row['月份']):02d}", axis=1)
                all_results.append(monthly_results)

        if not has_data_for_location:
            missing_data_sources.append(location)
            
    if not all_results:
        return pd.DataFrame(), missing_data_sources
        
    return pd.concat(all_results, ignore_index=True), missing_data_sources

@st.cache_data(ttl=3600, show_spinner="正在載入並預處理數據...")
def load_data(station_id, param_info_map):
    station_name = get_station_name_from_id(station_id)

    # 使用 st.expander 將所有的載入訊息包裹起來
    with st.expander(f"查看測站 '{station_name}' 的數據載入日誌"):
        st.info(f"嘗試從基本路徑 `{st.session_state.base_data_path}` 載入測站 `{station_name}` 的數據。")
        station_data_path = os.path.join(st.session_state.base_data_path, station_id)

        all_dfs = []
        found_any_file = False

        csv_files = glob(os.path.join(station_data_path, '*.csv')) + glob(os.path.join(station_data_path, '*.CSV'))
        if csv_files:
            st.info(f"在 `{station_data_path}` 中找到 {len(csv_files)} 個 CSV 檔案。")
            found_any_file = True
            for file_path in sorted(csv_files):
                try:
                    encodings = ['utf-8', 'latin1', 'big5', 'cp950']
                    df_part = None
                    for enc in encodings:
                        try:
                            df_part = pd.read_csv(file_path, header=1, encoding=enc, engine='python')
                            break
                        except UnicodeDecodeError:
                            continue
                    if df_part is None:
                        st.warning(f"文件 '{file_path}' 無法使用常見編碼解析。跳過此文件。")
                        continue

                    possible_time_cols = ['Time', 'time', 'UTC', 'GMT', 'Local_Time', 'Date', 'DateTime', 'TIME_UTC', 'Time (UTC)', 'time(UTC)', 'Time (LST)']
                    
                    # 清理列名，確保匹配時不會因為空格或大小寫問題錯過
                    df_part.columns = df_part.columns.str.strip().str.lower()
                    actual_time_cols_in_df = [col for col in df_part.columns if col in [pc.lower() for pc in possible_time_cols]]
                    
                    # --- 修正點 1: 更魯棒的日期時間解析 ---
                    # 定義多種可能的日期時間格式
                    possible_date_formats = [
                        '%Y/%m/%d %H:%M:%S', '%Y-%m-%d %H:%M:%S', 
                        '%Y/%m/%d %H:%M', '%Y-%m-%d %H:%M',      
                        '%Y/%m/%d', '%Y-%m-%d',                  
                        '%m/%d/%Y %H:%M:%S', '%d-%m-%Y %H:%M:%S', 
                        '%m/%d/%Y %H:%M', '%d-%m-%Y %H:%M',      
                        '%m/%d/%Y', '%d-%m-%Y',                  
                        '%Y%m%d%H%M%S', 
                        '%Y%m%d'      
                    ]
                    
                    found_time_col_and_parsed = False
                    for col in actual_time_cols_in_df:
                        # 嘗試使用明確格式解析
                        for fmt in possible_date_formats:
                            parsed_dates = pd.to_datetime(df_part[col], format=fmt, errors='coerce')
                            valid_time_ratio = parsed_dates.count() / len(df_part) if len(df_part) > 0 else 0
                            if valid_time_ratio > 0.5: # 如果超過一半的日期成功解析
                                df_part['ds'] = parsed_dates
                                found_time_col_and_parsed = True
                                st.info(f"文件 '{file_path}' 的時間列 '{col}' 已使用格式 '{fmt}' 成功解析。")
                                break 
                        if found_time_col_and_parsed:
                            break 

                    if not found_time_col_and_parsed:
                        # 如果所有明確格式都失敗，最後嘗試自動推斷（可能產生 UserWarning）
                        for col in actual_time_cols_in_df:
                            parsed_dates = pd.to_datetime(df_part[col], errors='coerce', infer_datetime_format=True)
                            valid_time_ratio = parsed_dates.count() / len(df_part) if len(df_part) > 0 else 0
                            if valid_time_ratio > 0.5:
                                df_part['ds'] = parsed_dates
                                found_time_col_and_parsed = True
                                st.warning(f"文件 '{file_path}' 的時間列 '{col}' 無法從預設格式中解析，已嘗試自動推斷格式 (可能較慢)。")
                                break
                            
                    if not found_time_col_and_parsed or df_part['ds'].isnull().all():
                        st.warning(f"文件 '{file_path}' 中未找到有效的時間列或時間格式無法解析。跳過此文件。")
                        continue
                    
                    df_part.set_index('ds', inplace=True)
                    all_dfs.append(df_part)
                except Exception as e:
                    st.warning(f"載入或處理文件 '{file_path}' 時發生錯誤：{e}。跳過此文件。")
                    continue

        if not found_any_file:
            st.error(f"錯誤：在測站 '{station_name}' 的任何指定子文件夾中都沒有找到有效的數據文件。")
            st.info(f"預期的測站數據根路徑: `{station_data_path}`")
            return pd.DataFrame()

        if not all_dfs:
            st.error(f"錯誤：雖然找到了 CSV 檔案，但沒有任何檔案成功載入並解析出有效時間序列數據。")
            return pd.DataFrame()

    # 合併所有 DataFrame，並移除重複索引
    combined_df = pd.concat(all_dfs).sort_index()
    combined_df = combined_df[~combined_df.index.duplicated(keep='first')]

    cleaned_df = combined_df.copy() 

    final_cols_to_keep = []
    # 遍歷參數資訊映射，找出要保留的列
    for param_key, param_info in param_info_map.items():
        param_col_in_data = param_info.get("column_name_in_data", param_key).lower()
        if param_col_in_data in cleaned_df.columns:
            # 嘗試轉換為數字，處理非數值數據
            cleaned_df[param_col_in_data] = pd.to_numeric(cleaned_df[param_col_in_data], errors='coerce')
            valid_ratio = cleaned_df[param_col_in_data].count() / len(cleaned_df) if len(cleaned_df) > 0 else 0

            # 根據參數類型和有效數據比例決定是否保留
            if param_info.get("type") in ["linear", "circular"] and valid_ratio > 0.1: # 至少10%的有效數據
                final_cols_to_keep.append(param_col_in_data)
            else:
                st.info(f"列 '{param_key}' (顯示名稱: {param_info.get('display_zh', 'N/A')}) 因數據類型不符、空值過多 ({valid_ratio*100:.2f}%) 或未配置為線性/圓形類型而被排除在主要分析之外。")
        else:
            st.info(f"配置文件中的參數 '{param_info.get('display_zh', param_key)}' (原始列名: '{param_key}') 未在數據文件中找到。")
    
    # 檢查 final_cols_to_keep 是否有內容
    if not final_cols_to_keep:
        st.warning(f"警告：測站 '{station_name}' 沒有符合條件的數據列用於分析。請檢查 config.json 中參數配置和數據內容。")
        return pd.DataFrame()

    # 僅選擇 final_cols_to_keep 中的欄位，索引 'ds' 會自動被保留
    cleaned_df = cleaned_df[final_cols_to_keep]

    # 確保最終 DataFrame 不是空的
    if cleaned_df.empty:
        st.error(f"錯誤：選擇參數後，數據為空。請檢查原始文件內容和列名是否與 config.json 匹配。")
        return pd.DataFrame()
    
    # 最終返回前重置索引，方便後續處理（如果你需要 'ds' 再次作為一個常規列）
    cleaned_df.reset_index(inplace=True) 

    return cleaned_df

def get_station_from_id(station_id):
    station_map = {d['StationID']: d for d in st.session_state['devices']}
    return station_map.get(station_id, station_id)

def get_station_name_from_id(station_id):
    return get_station_from_id(station_id).get('Title', station_id)

def analyze_navigability(df, wave_threshold, wind_threshold):
    if df is None or df.empty or 'Wave_Height_Significant' not in df.columns or 'Wind_Speed' not in df.columns: return np.nan
    
    df_navi = df[['Wave_Height_Significant', 'Wind_Speed']].copy()
    df_navi['Wave_Height_Significant'] = pd.to_numeric(df_navi['Wave_Height_Significant'], errors='coerce')
    df_navi['Wind_Speed'] = pd.to_numeric(df_navi['Wind_Speed'], errors='coerce')
    df_navi.dropna(inplace=True)
    if df_navi.empty: return np.nan
    
    navigable_conditions = df_navi[(df_navi['Wave_Height_Significant'] < wave_threshold) & (df_navi['Wind_Speed'] < wind_threshold)]
    return (len(navigable_conditions) / len(df_navi)) * 100

# --- 初始化 Session State，讓所有頁面能共享資料 ---
def initialize_session_state():
    if "initialized" in st.session_state and st.session_state.initialized:
        return

    st.session_state.initialized = True

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

    # 直接使用從 helpers 匯入的全局變數來初始化 Session State
    if 'base_data_path' not in st.session_state:
        st.session_state.base_data_path = BASE_DATA_PATH_FROM_CONFIG
    if 'chinese_font_path' not in st.session_state:
        st.session_state.chinese_font_path = CHINESE_FONT_PATH_FULL
    if 'chinese_font_name' not in st.session_state: # 新增：Plotly 需要字體名稱
        st.session_state.chinese_font_name = CHINESE_FONT_NAME
    if 'devices' not in st.session_state:
        with open(os.path.join(BASE_DATA_PATH_FROM_CONFIG, "devices.json"), 'r', encoding='utf-8') as f:
            st.session_state.devices = json.load(f)
    if 'parameter_info' not in st.session_state:
        st.session_state.parameter_info = PARAMETER_INFO
    if 'data_subfolders_priority' not in st.session_state:
        st.session_state.data_subfolders_priority = DATA_SUBFOLDERS_PRIORITY
    if 'locations' not in st.session_state:
        st.session_state.locations = [device['StationID'] for device in st.session_state.devices if 'StationID' in device]
    if 'risk_thresholds' not in st.session_state:
        st.session_state.risk_thresholds = RISK_THRESHOLDS

    if 'available_years' not in st.session_state:
        if st.session_state.locations and st.session_state.base_data_path:
            # get_available_years 會自行處理相對路徑轉換為絕對路徑
            st.session_state.available_years = get_available_years(st.session_state.base_data_path, st.session_state.locations)
        else:
            st.session_state.available_years = []

    # 應用 Matplotlib 中文字體設定 (如果需要 Matplotlib 圖表)
    # 如果 main app.py 頁面不直接繪製 Matplotlib 圖，這行程式碼也可以移動到需要繪圖的子頁面中。
    if st.session_state.chinese_font_path and st.session_state.chinese_font_name:
        set_chinese_font_for_matplotlib(st.session_state.chinese_font_path, st.session_state.chinese_font_name)
