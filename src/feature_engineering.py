# src/feature_engineering.py

import pandas as pd
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_DIR / "data" / "processed" / "cleaned_data.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "featured_data.csv"


def create_features():

    df = pd.read_csv(INPUT_PATH)

    # Convert date
    df["Date"] = pd.to_datetime(df["Date"])

    # Sort properly
    df = df.sort_values(["City", "Date"])

    # ---- Time Features ----
    df["month"] = df["Date"].dt.month
    df["day_of_week"] = df["Date"].dt.dayofweek

    # ---- Lag Features (per city) ----
    for lag in [1, 2, 3, 7]:
        df[f"AQI_lag_{lag}"] = df.groupby("City")["AQI"].shift(lag)

    # ---- Rolling Features ----
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

    # ---- Target Shift (Next Day Prediction) ----
    df["AQI_target"] = df.groupby("City")["AQI"].shift(-1)

    # Drop rows with NaN caused by shifting
    df = df.dropna()

    # Save
    os.makedirs(BASE_DIR / "data" / "processed", exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("Feature engineering completed.")
    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    create_features()
