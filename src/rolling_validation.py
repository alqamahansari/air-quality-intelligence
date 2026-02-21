import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "featured_data.csv"

def rolling_validation():

    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df["year"] = df["Date"].dt.year

    df_model = pd.get_dummies(df, columns=["City"], drop_first=True)

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
        "wind_speed",
    ] + [c for c in df_model.columns if c.startswith("City_")]

    splits = [
        (2021, 2020),
        (2022, 2021),
        (2023, 2022)
    ]

    print("\nRolling Temporal Validation:\n")

    for test_year, train_until in splits:

        train = df_model[df_model["year"] <= train_until]
        test = df_model[df_model["year"] == test_year]

        X_train = train[features]
        y_train = train["AQI_target"]
        X_test = test[features]
        y_test = test["AQI_target"]

        model = XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            random_state=42
        )

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)

        print(f"Train ≤ {train_until} → Test {test_year}")
        print(f"   R2  : {r2:.4f}")
        print(f"   MAE : {mae:.2f}\n")

if __name__ == "__main__":
    rolling_validation()