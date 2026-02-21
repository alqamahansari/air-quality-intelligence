# api/real_time_pipeline.py

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from api.fetch_api import fetch_pm25


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "featured_data.csv"
MODEL_PATH = BASE_DIR / "models" / "aqi_regressor.pkl"
FEATURE_PATH = BASE_DIR / "models" / "regression_features.pkl"
RESIDUAL_PATH = BASE_DIR / "models" / "residual_std.pkl"


reg_model = joblib.load(MODEL_PATH)
model_features = joblib.load(FEATURE_PATH)
residual_std = joblib.load(RESIDUAL_PATH)

df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])


def realtime_forecast(city_name, days=7):

    city_df = df[df["City"] == city_name].sort_values("Date").copy()

    if city_df.empty:
        return None

    last_row = city_df.iloc[-1:].copy()

    # ---- Fetch Real-Time PM2.5 ----
    realtime_pm25 = fetch_pm25(city_name)

    if realtime_pm25:
        # Replace lag_1 with real-time value
        last_row["AQI_lag_1"] = realtime_pm25

    predictions = []

    for _ in range(days):

        feature_row = last_row.copy()
        feature_row_encoded = pd.get_dummies(feature_row, columns=["City"], drop_first=True)

        for col in model_features:
            if col not in feature_row_encoded.columns:
                feature_row_encoded[col] = 0

        X_input = feature_row_encoded[model_features]

        pred = reg_model.predict(X_input)[0]

        lower = pred - 1.96 * residual_std
        upper = pred + 1.96 * residual_std

        predictions.append({
            "prediction": round(float(pred), 2),
            "lower_95": round(float(lower), 2),
            "upper_95": round(float(upper), 2)
        })

        # Update lag structure
        last_row["AQI_lag_1"] = pred

        last_row["Date"] = last_row["Date"] + pd.Timedelta(days=1)
        last_row["month"] = last_row["Date"].dt.month
        last_row["day_of_week"] = last_row["Date"].dt.dayofweek

    return predictions