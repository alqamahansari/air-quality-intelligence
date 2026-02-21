# src/multi_step_forecast.py

import pandas as pd
import numpy as np
import joblib
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "featured_data.csv"
MODEL_PATH = BASE_DIR / "models" / "aqi_regressor.pkl"
FEATURE_PATH = BASE_DIR / "models" / "regression_features.pkl"


def recursive_forecast(city_name, days=7):

    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])

    city_df = df[df["City"] == city_name].sort_values("Date").copy()
    last_row = city_df.iloc[-1:].copy()

    model = joblib.load(MODEL_PATH)
    model_features = joblib.load(FEATURE_PATH)

    predictions = []

    for step in range(days):

        feature_row = last_row.copy()

        # Encode city
        feature_row_encoded = pd.get_dummies(feature_row, columns=["City"], drop_first=True)

        # Add missing feature columns
        for col in model_features:
            if col not in feature_row_encoded.columns:
                feature_row_encoded[col] = 0

        # Ensure correct column order
        X_input = feature_row_encoded[model_features]

        pred = model.predict(X_input)[0]
        predictions.append(round(pred, 2))

        # Update lag structure
        last_row["AQI_lag_7"] = last_row["AQI_lag_3"]
        last_row["AQI_lag_3"] = last_row["AQI_lag_2"]
        last_row["AQI_lag_2"] = last_row["AQI_lag_1"]
        last_row["AQI_lag_1"] = pred

        last_row["AQI_roll_mean_3"] = np.mean([
            last_row["AQI_lag_1"].values[0],
            last_row["AQI_lag_2"].values[0],
            last_row["AQI_lag_3"].values[0]
        ])

        last_row["AQI_roll_mean_7"] = np.mean([
            last_row["AQI_lag_1"].values[0],
            last_row["AQI_lag_2"].values[0],
            last_row["AQI_lag_3"].values[0],
            last_row["AQI_lag_7"].values[0]
        ])

        # Update date
        last_row["Date"] = last_row["Date"] + pd.Timedelta(days=1)
        last_row["month"] = last_row["Date"].dt.month
        last_row["day_of_week"] = last_row["Date"].dt.dayofweek

    return predictions


if __name__ == "__main__":
    city = "Delhi"
    forecast = recursive_forecast(city, days=7)

    print(f"\n7-Day Forecast for {city}:")
    print(forecast)