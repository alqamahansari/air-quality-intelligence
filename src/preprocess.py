# src/preprocess.py

import pandas as pd
import glob
import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = BASE_DIR / "data" / "raw"
PROCESSED_PATH = BASE_DIR / "data" / "processed"

WEATHER_PATH = RAW_PATH / "weather_data.csv"


def preprocess_data():

    # -------- Load AQI CSV Files --------
    files = glob.glob(str(RAW_PATH / "*AQI*.csv"))

    df_list = []

    for file in files:
        temp = pd.read_csv(file)
        df_list.append(temp)

    aqi_df = pd.concat(df_list, ignore_index=True)

    # Remove unwanted columns
    aqi_df = aqi_df.loc[:, ~aqi_df.columns.str.contains("^Unnamed")]

    # Robust date parsing
    aqi_df["Date"] = pd.to_datetime(
        aqi_df["Date"],
        format="mixed",
        dayfirst=True,
        errors="coerce"
    )

    aqi_df = aqi_df.dropna(subset=["Date"])
    aqi_df = aqi_df.sort_values(["City", "Date"])

    print("AQI shape before weather merge:", aqi_df.shape)

    # -------- Load Weather Data --------
    if WEATHER_PATH.exists():

        weather_df = pd.read_csv(WEATHER_PATH)

        weather_df["Date"] = pd.to_datetime(
            weather_df["Date"],
            format="mixed",
            errors="coerce"
        )

        weather_df = weather_df.dropna(subset=["Date"])

        print("Weather shape:", weather_df.shape)

        # -------- Merge AQI + Weather --------
        merged_df = aqi_df.merge(
            weather_df,
            on=["City", "Date"],
            how="left"
        )

        print("After merge shape:", merged_df.shape)

    else:
        print("Weather file not found. Skipping weather merge.")
        merged_df = aqi_df

    # -------- Handle Missing Weather Values --------
    weather_cols = [
        "temp_max",
        "temp_min",
        "precipitation",
        "wind_speed"
    ]

    for col in weather_cols:
        if col in merged_df.columns:
            merged_df[col] = merged_df.groupby("City")[col].transform(
                lambda x: x.fillna(method="ffill").fillna(method="bfill")
            )

    # Final sort
    merged_df = merged_df.sort_values(["City", "Date"])

    os.makedirs(PROCESSED_PATH, exist_ok=True)
    merged_df.to_csv(PROCESSED_PATH / "combined_clean.csv", index=False)

    print("Preprocessing completed with weather integration.")
    print("Saved to:", PROCESSED_PATH / "combined_clean.csv")


if __name__ == "__main__":
    preprocess_data()