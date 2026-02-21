# src/explain_model.py

import pandas as pd
import shap
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "featured_data.csv"
MODEL_PATH = BASE_DIR / "models" / "aqi_regressor.pkl"
OUTPUT_PATH = BASE_DIR / "models" / "shap_summary.png"


def explain_model():

    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])

    df_model = pd.get_dummies(df, columns=["City"], drop_first=True)

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
    ] + [col for col in df_model.columns if col.startswith("City_")]

    X_test = test[features]

    model = joblib.load(MODEL_PATH)

    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)

    # Compute mean absolute SHAP importance
    mean_importance = np.abs(shap_values.values).mean(axis=0)

    importance_df = pd.DataFrame({
        "Feature": features,
        "Mean_SHAP_Importance": mean_importance
    }).sort_values(by="Mean_SHAP_Importance", ascending=False)

    print("\nTop 10 Important Features:\n")
    print(importance_df.head(10))

    # Save SHAP plot
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH)
    print(f"\nSHAP summary plot saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    explain_model()