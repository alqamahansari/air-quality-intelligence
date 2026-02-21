import pandas as pd
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "featured_data.csv"

def ablation():

    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])

    df_model = pd.get_dummies(df, columns=["City"], drop_first=True)

    train = df_model[df_model["Date"] < "2023-01-01"]
    test = df_model[df_model["Date"] >= "2023-01-01"]

    base_features = [
        "AQI_lag_1","AQI_lag_2","AQI_lag_3","AQI_lag_7",
        "AQI_roll_mean_3","AQI_roll_mean_7",
        "month","day_of_week"
    ] + [c for c in df_model.columns if c.startswith("City_")]

    variants = {
        "Full Model": base_features,
        "No Rolling Means": [f for f in base_features if "roll" not in f],
        "No City Encoding": [f for f in base_features if not f.startswith("City_")],
        "No Temporal Features": [f for f in base_features if f not in ["month","day_of_week"]]
    }

    print("\nFeature Ablation Study:\n")

    for name, features in variants.items():

        model = XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            random_state=42
        )

        model.fit(train[features], train["AQI_target"])
        preds = model.predict(test[features])
        r2 = r2_score(test["AQI_target"], preds)

        print(f"{name}: R2 = {r2:.4f}")

if __name__ == "__main__":
    ablation()