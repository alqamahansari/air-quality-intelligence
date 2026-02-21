# src/train_regression.py

import pandas as pd
import numpy as np
import os
from pathlib import Path
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "featured_data.csv"
MODEL_PATH = BASE_DIR / "models" / "aqi_regressor.pkl"
FEATURE_PATH = BASE_DIR / "models" / "regression_features.pkl"
RESIDUAL_PATH = BASE_DIR / "models" / "residual_std.pkl"


def train_model():

    # Load data
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])

    # Preserve original city column for per-city evaluation
    df_original = df.copy()

    # Encode city for modeling
    df_model = pd.get_dummies(df, columns=["City"], drop_first=True)

    # Time-based split
    train = df_model[df_model["Date"] < "2023-01-01"]
    test = df_model[df_model["Date"] >= "2023-01-01"]

    features = [
        "AQI_lag_1",
        "AQI_lag_2",
        "AQI_lag_3",
        "AQI_lag_7",
        "AQI_roll_mean_3",
        "AQI_roll_mean_7",
        "month",
        "day_of_week",
        "temp_max",
        "temp_min",
        "precipitation",
        "wind_speed"
    ] + [col for col in df_model.columns if col.startswith("City_")]

    X_train = train[features]
    y_train = train["AQI_target"]

    X_test = test[features]
    y_test = test["AQI_target"]

    # Model
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # -----------------------------
    # Overall Metrics
    # -----------------------------
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(((y_test - y_pred) ** 2).mean())
    r2 = r2_score(y_test, y_pred)

    print("\nOverall Model Performance:")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R2   : {r2:.4f}")

    # -----------------------------
    # Residual Standard Deviation (Uncertainty)
    # -----------------------------
    residuals = y_test - y_pred
    residual_std = np.std(residuals)

    print(f"\nResidual Std Dev (for 95% CI): {residual_std:.2f}")

    # -----------------------------
    # Naïve Persistence Baseline
    # -----------------------------
    baseline_pred = test["AQI_lag_1"].values

    baseline_mae = mean_absolute_error(y_test, baseline_pred)
    baseline_rmse = np.sqrt(((y_test - baseline_pred) ** 2).mean())
    baseline_r2 = r2_score(y_test, baseline_pred)

    print("\nNaïve Persistence Baseline Performance:")
    print(f"MAE  : {baseline_mae:.2f}")
    print(f"RMSE : {baseline_rmse:.2f}")
    print(f"R2   : {baseline_r2:.4f}")

    improvement = (r2 - baseline_r2) * 100
    print(f"\nR2 Improvement over Baseline: {improvement:.2f}%")

    # -----------------------------
    # Per-City Evaluation
    # -----------------------------
    print("\n--- Per-City Performance ---\n")

    test_original = df_original[df_original["Date"] >= "2023-01-01"].copy()
    test_original = test_original.reset_index(drop=True)
    test_original["Prediction"] = y_pred

    for city in test_original["City"].unique():

        city_data = test_original[test_original["City"] == city]

        city_r2 = r2_score(city_data["AQI_target"], city_data["Prediction"])
        city_mae = mean_absolute_error(city_data["AQI_target"], city_data["Prediction"])

        print(f"{city}")
        print(f"   R2  : {city_r2:.4f}")
        print(f"   MAE : {city_mae:.2f}\n")

    # -----------------------------
    # Save Everything
    # -----------------------------
    os.makedirs(BASE_DIR / "models", exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(features, FEATURE_PATH)
    joblib.dump(residual_std, RESIDUAL_PATH)

    print("Model saved to:", MODEL_PATH)
    print("Residual std saved to:", RESIDUAL_PATH)


if __name__ == "__main__":
    train_model()