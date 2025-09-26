import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Step 1: Load data
data = [
    {"date": "2025-09-20", "max_temp_C": 26, "min_temp_C": 20, "avg_humidity": 90, "rainfall_mm": 16.8},
    {"date": "2025-09-21", "max_temp_C": 27, "min_temp_C": 21, "avg_humidity": 86, "rainfall_mm": 4.8},
    {"date": "2025-09-22", "max_temp_C": 26, "min_temp_C": 20, "avg_humidity": 87, "rainfall_mm": 3.0},
    {"date": "2025-09-23", "max_temp_C": 22, "min_temp_C": 19, "avg_humidity": 93, "rainfall_mm": 13.2},
    {"date": "2025-09-24", "max_temp_C": 27, "min_temp_C": 19, "avg_humidity": 85, "rainfall_mm": 0.7},
    {"date": "2025-09-25", "max_temp_C": 28, "min_temp_C": 19, "avg_humidity": 83, "rainfall_mm": 0.0},
    {"date": "2025-09-26", "max_temp_C": 26, "min_temp_C": 20, "avg_humidity": 86, "rainfall_mm": 0.0}
]

df = pd.DataFrame(data)
df['day_index'] = range(1, len(df)+1)

# Step 2: Features & targets
X = df[['day_index']]
y_max = df['max_temp_C']
y_min = df['min_temp_C']
y_rain = df['rainfall_mm']
y_hum = df['avg_humidity']

# Step 3: Train models
model_max = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y_max)
model_min = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y_min)
model_rain = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y_rain)
model_hum = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y_hum)

# Step 4: Predict next 7 days
future_days = pd.DataFrame({'day_index': range(len(df)+1, len(df)+8)})
pred_max = model_max.predict(future_days)
pred_min = model_min.predict(future_days)
pred_rain = model_rain.predict(future_days)
pred_hum = model_hum.predict(future_days)

# Step 5: Combine predictions
forecast = pd.DataFrame({
    'day_index': range(len(df)+1, len(df)+8),
    'pred_max_temp': pred_max,
    'pred_min_temp': pred_min,
    'pred_rainfall': pred_rain,
    'pred_humidity': pred_hum
})

print(forecast)
