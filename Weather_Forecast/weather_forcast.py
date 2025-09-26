import pandas as pd
from prophet import Prophet
import json
import os

# --------------------------
# Config
# --------------------------
JSON_PATH = "Weather_Forecast/weather_data.json"
OUTPUT_PATH = "Weather_Forecast/weather_forecast_7days.csv"
FORECAST_DAYS = 7
LOCATION = "Pune"

# --------------------------
# Load data
# --------------------------
with open(JSON_PATH, "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# Convert columns to numeric
df['max_temp_C'] = pd.to_numeric(df['max_temp_C'])
df['min_temp_C'] = pd.to_numeric(df['min_temp_C'])
df['rainfall_mm'] = pd.to_numeric(df['rainfall_mm'])
df['avg_humidity'] = pd.to_numeric(df['avg_humidity'])

# Prepare dataframe for Prophet
def prepare_prophet_df(series, col_name):
    return pd.DataFrame({
        'ds': pd.to_datetime(df['date']),
        'y': series
    })

# --------------------------
# Train Prophet models
# --------------------------
def train_prophet(series):
    m = Prophet(daily_seasonality=True, yearly_seasonality=True)
    m.fit(series)
    future = m.make_future_dataframe(periods=FORECAST_DAYS)
    forecast = m.predict(future)
    return forecast[['ds', 'yhat']].tail(FORECAST_DAYS)['yhat'].values

pred_max = train_prophet(prepare_prophet_df(df['max_temp_C'], 'max_temp_C'))
pred_min = train_prophet(prepare_prophet_df(df['min_temp_C'], 'min_temp_C'))
pred_rain = train_prophet(prepare_prophet_df(df['rainfall_mm'], 'rainfall_mm'))
pred_hum = train_prophet(prepare_prophet_df(df['avg_humidity'], 'avg_humidity'))

# --------------------------
# Clip unrealistic values
# --------------------------
pred_rain = [max(0, r) for r in pred_rain]
pred_hum = [min(max(0, h), 100) for h in pred_hum]  # 0-100%
pred_max = [round(m, 1) for m in pred_max]
pred_min = [round(m, 1) for m in pred_min]
pred_rain = [round(r, 1) for r in pred_rain]
pred_hum = [round(h, 1) for h in pred_hum]

# --------------------------
# Build forecast output
# --------------------------
last_date = pd.to_datetime(df['date']).max()
forecast_dates = [last_date + pd.Timedelta(days=i+1) for i in range(FORECAST_DAYS)]

forecast = []
for i in range(FORECAST_DAYS):
    forecast.append({
        "date": forecast_dates[i].strftime("%Y-%m-%d"),
        "pred_max_temp": pred_max[i],
        "pred_min_temp": pred_min[i],
        "pred_rainfall": pred_rain[i],
        "pred_humidity": pred_hum[i]
    })

# Save to JSON
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    pd.DataFrame(forecast).to_csv(f, index=False)

print(f"7-day forecast saved to {OUTPUT_PATH}")
