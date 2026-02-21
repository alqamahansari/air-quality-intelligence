# src/evaluate_multi_step.py

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "featured_data.csv"
MODEL_PATH = BASE_DIR / "models" / "aqi_regressor.pkl"
FEATURE_PATH = BASE_DIR / "models" / "regression_features.pkl"


def evaluate_multi_step(days=7):

    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])

    model = joblib.load(MODEL_PATH)
    model_features = joblib.load(FEATURE_PATH)

    test_df = df[df["Date"] >= "2023-01-01"].copy()

    results = {i: {"true": [], "pred": []} for i in range(1, days+1)}

    for idx in range(len(test_df) - days):

        row = test_df.iloc[idx:idx+1].copy()
        city = row["City"].values[0]

        future_rows = test_df[
            (test_df["City"] == city)
        ].iloc[idx+1:idx+1+days]

        if len(future_rows) < days:
            continue

        last_row = row.copy()

        for step in range(1, days+1):

            feature_row = last_row.copy()
            feature_row_encoded = pd.get_dummies(feature_row, columns=["City"], drop_first=True)

            for col in model_features:
                if col not in feature_row_encoded.columns:
                    feature_row_encoded[col] = 0

            X_input = feature_row_encoded[model_features]
            pred = model.predict(X_input)[0]

            true_value = future_rows.iloc[step-1]["AQI"]

            results[step]["true"].append(true_value)
            results[step]["pred"].append(pred)

            # update lags
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

    print("\nMulti-Step Forecast Evaluation:\n")

    for step in range(1, days+1):

        r2 = r2_score(results[step]["true"], results[step]["pred"])
        mae = mean_absolute_error(results[step]["true"], results[step]["pred"])

        print(f"Horizon t+{step}")
        print(f"   R2  : {r2:.4f}")
        print(f"   MAE : {mae:.2f}\n")


if __name__ == "__main__":
    evaluate_multi_step(days=7)