import requests
import json
import os
from datetime import datetime, timedelta

API_KEY = "d68eeba563b342a890460420252609"
BASE_URL = "http://api.worldweatheronline.com/premium/v1/past-weather.ashx"
OUTPUT_PATH = "Weather_Forecast/weather_data.json"

def get_past_weather(location, start_date, end_date):
    """Fetch past weather data for a given location and date range."""
    url = f"{BASE_URL}?key={API_KEY}&q={location}&date={start_date}&enddate={end_date}&tp=24&format=json"
    response = requests.get(url)
    data = response.json()
    
    history = []
    if 'data' in data and 'weather' in data['data']:
        for d in data['data']['weather']:
            history.append({
                "date": d["date"],
                "max_temp_C": d["maxtempC"],
                "min_temp_C": d["mintempC"],
                "avg_humidity": d["hourly"][0]["humidity"],
                "rainfall_mm": d["hourly"][0]["precipMM"]
            })
    else:
        print(f"No data returned for {start_date} to {end_date}")
    
    return history

def fetch_weather_for_year(location="Pune", year=2025, chunk_days=7):
    """Fetch past weather data for an entire year in chunks."""
    start_dt = datetime(year, 1, 1)
    end_dt = datetime(year, 12, 31)
    
    all_history = []
    while start_dt <= end_dt:
        chunk_end = start_dt + timedelta(days=chunk_days-1)
        if chunk_end > end_dt:
            chunk_end = end_dt
        
        start_str = start_dt.strftime("%Y-%m-%d")
        end_str = chunk_end.strftime("%Y-%m-%d")
        
        history = get_past_weather(location, start_str, end_str)
        all_history.extend(history)
        
        start_dt = chunk_end + timedelta(days=1)
    
    # Save to JSON
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_history, f, indent=4)
    
    print(f"Saved {len(all_history)} days of weather data for {location}, {year}.")
    return all_history

# Example usage
if __name__ == "__main__":
    fetch_weather_for_year("Pune", 2025)
