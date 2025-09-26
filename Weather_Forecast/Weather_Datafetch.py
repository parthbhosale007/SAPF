import requests
import json

API_KEY = "d68eeba563b342a890460420252609"
BASE_URL = "http://api.worldweatheronline.com/premium/v1/past-weather.ashx"
output_path = "Weather_Forecast/weather_data.json"

def get_past_weather(location="Pune", start_date="2025-09-20", end_date="2025-09-26"):
    url = f"{BASE_URL}?key={API_KEY}&q={location}&date={start_date}&enddate={end_date}&tp=24&format=json"
    response = requests.get(url)
    data = response.json()
    
    history = []
    for d in data['data']['weather']:
        history.append({
            "date": d["date"],
            "max_temp_C": d["maxtempC"],
            "min_temp_C": d["mintempC"],
            "avg_humidity": d["hourly"][0]["humidity"],
            "rainfall_mm": d["hourly"][0]["precipMM"]
        })
    
    return history

# Example
if __name__ == "__main__":
    result = get_past_weather("Pune", "2025-09-20", "2025-09-26")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
    json.dump(result, indent=4)
    # print(json.dumps(result, indent=4))
