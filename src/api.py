# src/api.py

import asyncio
import requests
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from fastapi import FastAPI, WebSocket, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
# =========================
# PATHS
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "featured_data.csv"
REG_MODEL_PATH = BASE_DIR / "models" / "aqi_regressor.pkl"
CLS_MODEL_PATH = BASE_DIR / "models" / "aqi_classifier.pkl"
FEATURE_PATH = BASE_DIR / "models" / "regression_features.pkl"
RESIDUAL_PATH = BASE_DIR / "models" / "residual_std.pkl"


# =========================
# APP SETUP
# =========================

app = FastAPI(title="Air Quality Intelligence API")

app.mount("/static", StaticFiles(directory="src/static"), name="static")
templates = Jinja2Templates(directory="src/templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# LOAD MODELS
# =========================

reg_model = joblib.load(REG_MODEL_PATH)
cls_model = joblib.load(CLS_MODEL_PATH)
model_features = joblib.load(FEATURE_PATH)
residual_std = joblib.load(RESIDUAL_PATH)

df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])


# =========================
# REAL-TIME API FETCH
# =========================

def fetch_realtime_pm25(city_name: str):

    city_map = {
        "Delhi": "Delhi",
        "Mumbai": "Mumbai",
        "Bengaluru": "Bengaluru",
        "Chennai": "Chennai",
        "Hyderabad": "Hyderabad"
    }

    url = "https://api.openaq.org/v2/latest"

    params = {
        "city": city_map.get(city_name, city_name),
        "parameter": "pm25",
        "limit": 1
    }

    try:
        response = requests.get(url, params=params, timeout=5)

        if response.status_code != 200:
            return None

        data = response.json()

        if not data["results"]:
            return None

        value = data["results"][0]["measurements"][0]["value"]
        return float(value)

    except:
        return None


# =========================
# PM2.5 → AQI (Approx CPCB)
# =========================

def pm25_to_aqi(pm25):

    if pm25 <= 30:
        return 50
    elif pm25 <= 60:
        return 100
    elif pm25 <= 90:
        return 200
    elif pm25 <= 120:
        return 300
    elif pm25 <= 250:
        return 400
    else:
        return 500


# =========================
# HEALTH LABEL
# =========================

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


# =========================
# SINGLE STEP (LIVE)
# =========================

def single_step_prediction(city_name: str):

    city_df = df[df["City"] == city_name].sort_values("Date").copy()

    if city_df.empty:
        return None

    last_row = city_df.iloc[-1:].copy()

    # ---- Inject real-time data ----
    realtime_pm25 = fetch_realtime_pm25(city_name)

    if realtime_pm25:
        realtime_aqi = pm25_to_aqi(realtime_pm25)
        last_row["AQI_lag_1"] = realtime_aqi

    # ---- Encode features ----
    feature_row_encoded = pd.get_dummies(last_row, columns=["City"], drop_first=True)

    for col in model_features:
        if col not in feature_row_encoded.columns:
            feature_row_encoded[col] = 0

    X_input = feature_row_encoded[model_features]

    # ---- Regression ----
    pred = float(reg_model.predict(X_input)[0])
    lower = pred - 1.96 * residual_std
    upper = pred + 1.96 * residual_std

    # ---- Classification ----
    cls_pred = cls_model.predict(X_input)[0]
    category = categorize_label(cls_pred)

    return {
        "aqi": round(pred, 2),
        "lower_95": round(float(lower), 2),
        "upper_95": round(float(upper), 2),
        "health_risk": category,
        "timestamp": pd.Timestamp.utcnow().isoformat()
    }


# =========================
# RECURSIVE FORECAST
# =========================

def recursive_forecast(city_name, days=7):

    city_df = df[df["City"] == city_name].sort_values("Date").copy()

    if city_df.empty:
        return None

    last_row = city_df.iloc[-1:].copy()
    forecast_output = []

    for _ in range(days):

        feature_row_encoded = pd.get_dummies(last_row, columns=["City"], drop_first=True)

        for col in model_features:
            if col not in feature_row_encoded.columns:
                feature_row_encoded[col] = 0

        X_input = feature_row_encoded[model_features]

        pred = float(reg_model.predict(X_input)[0])
        lower = pred - 1.96 * residual_std
        upper = pred + 1.96 * residual_std

        cls_pred = cls_model.predict(X_input)[0]
        category = categorize_label(cls_pred)

        forecast_output.append({
            "prediction": round(pred, 2),
            "lower_95": round(float(lower), 2),
            "upper_95": round(float(upper), 2),
            "health_risk": category
        })

        # ---- Update lags ----
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

        last_row["Date"] += pd.Timedelta(days=1)
        last_row["month"] = last_row["Date"].dt.month
        last_row["day_of_week"] = last_row["Date"].dt.dayofweek

    return forecast_output


# =========================
# DASHBOARD ROUTE
# =========================

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# =========================
# REST FORECAST ENDPOINT
# =========================

@app.get("/predict")
def predict(city: str = "Delhi"):

    forecast = recursive_forecast(city, days=7)

    if forecast is None:
        return {"error": "City not found"}

    return {
        "city": city,
        "7_day_forecast_with_uncertainty": forecast
    }


# =========================
# WEBSOCKET LIVE STREAM
# =========================

connected_clients = []
CITIES = ["Delhi", "Mumbai", "Bengaluru", "Chennai", "Hyderabad"]


@app.on_event("startup")
async def start_streaming():
    asyncio.create_task(stream_loop())


async def stream_loop():
    while True:

        live_data = {}

        for city in CITIES:
            result = single_step_prediction(city)
            if result:
                live_data[city] = result

        for client in connected_clients:
            await client.send_json(live_data)

        await asyncio.sleep(10)   # slower to avoid API overload


@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)

    try:
        while True:
            await websocket.receive_text()
    except:
        connected_clients.remove(websocket)


# =========================
# ROOT
# =========================

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/dashboard")