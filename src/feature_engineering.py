# src/feature_engineering.py

import pandas as pd
import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "combined_clean.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "featured_data.csv"


def create_features():

    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])

    # ---- Time features ----
    df["month"] = df["Date"].dt.month
    df["day_of_week"] = df["Date"].dt.dayofweek

    # ---- Lag features per city ----
    for lag in [1, 2, 3, 7]:
        df[f"AQI_lag_{lag}"] = df.groupby("City")["AQI"].shift(lag)

    # ---- Rolling features per city ----
    df["AQI_roll_mean_3"] = (
        df.groupby("City")["AQI"]
          .rolling(3)
          .mean()
          .reset_index(0, drop=True)
    )

    df["AQI_roll_mean_7"] = (
        df.groupby("City")["AQI"]
          .rolling(7)
          .mean()
          .reset_index(0, drop=True)
    )

    # ---- Forecast Target (Next Day AQI) ----
    df["AQI_target"] = df.groupby("City")["AQI"].shift(-1)

    # ---- Health Risk Category (Based on AQI_target) ----
    def categorize_aqi(aqi):
        if aqi <= 50:
            return 0  # Good
        elif aqi <= 100:
            return 1  # Satisfactory
        elif aqi <= 200:
            return 2  # Moderate
        elif aqi <= 300:
            return 3  # Poor
        elif aqi <= 400:
            return 4  # Very Poor
        else:
            return 5  # Severe

    df["AQI_Category"] = df["AQI_target"].apply(categorize_aqi)

    # ---- Drop rows only required for forecasting/classification ----
    forecast_cols = [
        "AQI_lag_1",
        "AQI_lag_2",
        "AQI_lag_3",
        "AQI_lag_7",
        "AQI_roll_mean_3",
        "AQI_roll_mean_7",
        "AQI_target",
        "AQI_Category"
    ]

    df = df.dropna(subset=forecast_cols)

    # Save
    os.makedirs(BASE_DIR / "data" / "processed", exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("Feature engineering completed.")
    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    create_features()