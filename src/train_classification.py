# src/train_classification.py

import pandas as pd
import numpy as np
import os
from pathlib import Path
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "featured_data.csv"
MODEL_PATH = BASE_DIR / "models" / "aqi_classifier.pkl"


def train_classifier():

    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])

    # Encode city
    df = pd.get_dummies(df, columns=["City"], drop_first=True)

    train = df[df["Date"] < "2023-01-01"]
    test = df[df["Date"] >= "2023-01-01"]

    features = [
        "AQI_lag_1",
        "AQI_lag_2",
        "AQI_lag_3",
        "AQI_lag_7",
        "AQI_roll_mean_3",
        "AQI_roll_mean_7",
        "month",
        "day_of_week",
    ] + [col for col in df.columns if col.startswith("City_")]

    X_train = train[features]
    y_train = train["AQI_Category"]

    X_test = test[features]
    y_test = test["AQI_Category"]

    # ---- Compute Class Weights ----
    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )
    class_weights = dict(zip(classes, weights))

    sample_weights = y_train.map(class_weights)

    # ---- Model ----
    model = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        objective="multi:softprob",
        num_class=6
    )

    model.fit(X_train, y_train, sample_weight=sample_weights)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("\nBalanced Classification Performance:")
    print("Accuracy:", round(accuracy, 4))
    print("\nDetailed Report:\n")
    print(classification_report(y_test, y_pred))

    os.makedirs(BASE_DIR / "models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print("\nClassifier saved to:", MODEL_PATH)


if __name__ == "__main__":
    train_classifier()