# src/preprocess.py

import pandas as pd
import os
from pathlib import Path

# Get project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

RAW_PATH = BASE_DIR / "data" / "raw" / "city.csv"
PROCESSED_PATH = BASE_DIR / "data" / "processed" / "cleaned_data.csv"


def preprocess_data():
    df = pd.read_csv(RAW_PATH)

    # Clean column names
    df.columns = (
        df.columns
          .str.strip()
          .str.replace(" ", "_")
    )

    # Convert Date
    df["Date"] = pd.to_datetime(df["Date"])

    # Sort
    df = df.sort_values(by=["City", "Date"])

    os.makedirs(BASE_DIR / "data" / "processed", exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)

    print("Preprocessing completed.")
    print("Saved to:", PROCESSED_PATH)


if __name__ == "__main__":
    preprocess_data()
