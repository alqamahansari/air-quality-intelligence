import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "featured_data.csv"

SEQ_LENGTH = 7
EPOCHS = 60
LR = 0.001


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length, 0])  # target is AQI
    return np.array(X), np.array(y)


def run_lstm():

    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])

    df = df[df["City"] == "Delhi"]
    df = df.sort_values("Date")

    features = [
        "AQI",
        "temp_max",
        "temp_min",
        "precipitation",
        "wind_speed"
    ]

    train_df = df[df["Date"] < "2023-01-01"]
    test_df = df[df["Date"] >= "2023-01-01"]

    scaler = MinMaxScaler()

    train_scaled = scaler.fit_transform(train_df[features])
    test_scaled = scaler.transform(test_df[features])

    X_train, y_train = create_sequences(train_scaled, SEQ_LENGTH)
    X_test, y_test = create_sequences(test_scaled, SEQ_LENGTH)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)

    model = LSTMModel(input_size=len(features))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(X_test).numpy()

    # Inverse scaling (only AQI column)
    preds_full = np.zeros((len(preds), len(features)))
    y_test_full = np.zeros((len(preds), len(features)))

    preds_full[:, 0] = preds.flatten()
    y_test_full[:, 0] = y_test.numpy().flatten()

    preds_inv = scaler.inverse_transform(preds_full)[:, 0]
    y_test_inv = scaler.inverse_transform(y_test_full)[:, 0]

    r2 = r2_score(y_test_inv, preds_inv)
    mae = mean_absolute_error(y_test_inv, preds_inv)

    print("\nMultivariate LSTM (Delhi):")
    print(f"R2  : {r2:.4f}")
    print(f"MAE : {mae:.2f}")


if __name__ == "__main__":
    run_lstm()