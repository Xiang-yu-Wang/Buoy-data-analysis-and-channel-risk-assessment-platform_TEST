import schedule
import time
import requests
import csv
from os import makedirs, mkdir, path
from datetime import datetime, timedelta

OUTPUT = "dataset/buoy/"
CSV_COLUMNS = [
    "StationID",
    "time",
    "Wind_Gust_Speed",
    "Wind_Speed",
    "Wind_Direction",
    "Air_Pressure",
    "Air_Temperature",
    "Sea_Temperature",
    "Wave_Height_Significant",
    "Wave_Mean_Period",
    "Wave_Main_Direction",
    "Wave_Peak_Period",
    "Current_Speed",
    "Current_Speed_Layer",
    "Current_Direction",
    "Current_Direction_Layer",
    "Current_Speed_knot",
    "Tide_Height"
]

CSV_CHINESE_COLUMNS = [
    "測站編號",
    "時間",
    "陣風_風速",
    "風速",
    "風向",
    "氣壓",
    "氣溫",
    "海面溫度",
    "示性波高",
    "平均週期",
    "波向",
    "波浪尖峰週期",
    "流速",
    "分層流速{深度:流速}",
    "流向",
    "分層流向{深度:流向}",
    "流速(節)",
    "潮高"
]

CSV_UNITS = [
    "",
    "UTC+8",
    "m/s",
    "m/s",
    "degree",
    "hPa",
    "C",
    "C",
    "m",
    "sec",
    "degree",
    "sec",
    "m/s",
    "m/s",
    "degree",
    "degree",
    "knot",
    "m"
]

def fetch_data(device_id: str):
    now = datetime.now()
    start_time = (now - timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%S")
    end_time = now.strftime("%Y-%m-%dT%H:%M:%S")

    # Format the URL with query parameters
    API_URL = f"https://nodass.namr.gov.tw/noapi/namr/v1/obs/{device_id}/data?date1={start_time}&date2={end_time}"

    print(f"[{now}] Requesting: {API_URL}")
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(API_URL, timeout=10)
            response.raise_for_status()
            print(f"✅ Success on attempt {attempt}")
            data = response.json()
            parse_to_csv(data, device_id)
            break
        except Exception as e:
            print(f"⚠️ Attempt {attempt} failed: {e}")
            if attempt < max_retries:
                time.sleep(5)
            else:
                print("❌ All retries failed.")


def parse_to_csv(data, device_id):
    rows = data if isinstance(data, list) else [data]
    filename = f"{datetime.now().strftime('%Y%m')}.csv"
    output = path.join(OUTPUT, device_id, filename)

    with open(output, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)

        if file.tell() == 0:
            # Header have 3 lines: Chinese names, English names, and units
            chinese_header = {key: value for key, value in zip(CSV_COLUMNS, CSV_CHINESE_COLUMNS)}
            writer.writerow(chinese_header)

            writer.writeheader()  # Write English header

            header_units = {key: value for key, value in zip(CSV_COLUMNS, CSV_UNITS)}
            writer.writerow(header_units)

        for row in rows:
            filtered = {key: row.get(key, "") for key in CSV_COLUMNS}
            writer.writerow(filtered)
            print(f"📝 Appended row: {filtered}")


def fetch_all_devices():
    device_ids = [
        "Vector_CWB_FB_46694A",
        "Vector_CWB_FB_46714D",
        "Vector_WRA_FB_46706A",
        "Vector_WRA_FB_46759A",
        "Vector_WRA_FB_WRA007",
        "Vector_WRA_FB_COMC08",
        "Vector_NAMR_FB_35A0004",
        "Vector_NAMR_FB_31A0005",
        "Vector_NAMR_FB_33A0006"
    ]
    for device_id in device_ids:
        makedirs(path.join(OUTPUT, device_id), exist_ok=True)
        fetch_data(device_id)

# Schedule every 2 days at 08:00
schedule.every(2).days.at("08:00").do(fetch_all_devices)

# Run immediately on startup
fetch_all_devices()
print("🔄 Scheduler started. Will run every 2 days.")
while True:
    schedule.run_pending()
    time.sleep(3600)
