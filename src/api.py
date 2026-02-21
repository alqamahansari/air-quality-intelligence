# src/api.py

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "featured_data.csv"
REG_MODEL_PATH = BASE_DIR / "models" / "aqi_regressor.pkl"
CLS_MODEL_PATH = BASE_DIR / "models" / "aqi_classifier.pkl"
FEATURE_PATH = BASE_DIR / "models" / "regression_features.pkl"
RESIDUAL_PATH = BASE_DIR / "models" / "residual_std.pkl"


app = FastAPI(title="Air Quality Intelligence API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models once at startup
reg_model = joblib.load(REG_MODEL_PATH)
cls_model = joblib.load(CLS_MODEL_PATH)
model_features = joblib.load(FEATURE_PATH)
residual_std = joblib.load(RESIDUAL_PATH)

df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])


def categorize_label(label):
    mapping = {
        0: "Good",
        1: "Satisfactory",
        2: "Moderate",
        3: "Poor",
        4: "Very Poor",
        5: "Severe"
    }
    return mapping.get(int(label), "Unknown")


def recursive_forecast(city_name, days=7):

    city_df = df[df["City"] == city_name].sort_values("Date").copy()

    if city_df.empty:
        return None

    last_row = city_df.iloc[-1:].copy()
    forecast_output = []

    for step in range(days):

        feature_row = last_row.copy()
        feature_row_encoded = pd.get_dummies(feature_row, columns=["City"], drop_first=True)

        # Ensure all required features exist
        for col in model_features:
            if col not in feature_row_encoded.columns:
                feature_row_encoded[col] = 0

        X_input = feature_row_encoded[model_features]

        # -------- Regression Prediction --------
        pred = float(reg_model.predict(X_input)[0])

        lower = pred - 1.96 * residual_std
        upper = pred + 1.96 * residual_std

        # -------- Classification Prediction --------
        cls_pred = cls_model.predict(X_input)[0]
        category = categorize_label(cls_pred)

        forecast_output.append({
            "prediction": round(pred, 2),
            "lower_95": round(float(lower), 2),
            "upper_95": round(float(upper), 2),
            "health_risk": category
        })

        # -------- Update Lag Structure --------
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

        last_row["Date"] = last_row["Date"] + pd.Timedelta(days=1)
        last_row["month"] = last_row["Date"].dt.month
        last_row["day_of_week"] = last_row["Date"].dt.dayofweek

    return forecast_output


@app.get("/")
def root():
    return {"message": "Air Quality Intelligence API Running"}


@app.get("/predict")
def predict(city: str = "Delhi"):

    forecast = recursive_forecast(city, days=7)

    if forecast is None:
        return {"error": "City not found"}

    return {
        "city": city,
        "7_day_forecast_with_uncertainty": forecast
    }