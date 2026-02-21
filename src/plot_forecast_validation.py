# src/plot_forecast_validation.py

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "featured_data.csv"
MODEL_PATH = BASE_DIR / "models" / "aqi_regressor.pkl"
FEATURE_PATH = BASE_DIR / "models" / "regression_features.pkl"
RESIDUAL_PATH = BASE_DIR / "models" / "residual_std.pkl"

OUTPUT_PATH = BASE_DIR / "models" / "forecast_validation_plot.png"


def plot_validation(city="Delhi"):

    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])

    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURE_PATH)
    residual_std = joblib.load(RESIDUAL_PATH)

    # Filter test period
    df = df[df["Date"] >= "2023-01-01"]
    df = df[df["City"] == city]

    # Encode
    df_encoded = pd.get_dummies(df, columns=["City"], drop_first=True)

    for col in features:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    X = df_encoded[features]
    y_true = df["AQI_target"].values

    y_pred = model.predict(X)

    lower = y_pred - 1.96 * residual_std
    upper = y_pred + 1.96 * residual_std

    # Plot
    plt.figure(figsize=(14,6))

    plt.plot(df["Date"], y_true, label="True AQI", linewidth=2)
    plt.plot(df["Date"], y_pred, label="Predicted AQI", linewidth=2)

    plt.fill_between(
        df["Date"],
        lower,
        upper,
        alpha=0.2,
        label="95% Confidence Interval"
    )

    plt.title(f"AQI Forecast Validation - {city}")
    plt.xlabel("Date")
    plt.ylabel("AQI")
    plt.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH)
    plt.close()

    print("Validation plot saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    plot_validation("Delhi")