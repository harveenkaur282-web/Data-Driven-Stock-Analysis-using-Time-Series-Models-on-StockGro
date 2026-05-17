from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.utils.metrics import evaluate

log = logging.getLogger(__name__)

FORECASTS_DIR = Path(__file__).resolve().parents[2] / "results" / "forecasts"
FORECASTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_sequences(data: np.ndarray, seq_len: int = 60) -> Tuple[np.ndarray, np.ndarray]:
#creates (X,y) sequences from 1-D scaled array.
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len: i])
        y.append(data[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

class _RNNBase(nn.Module):
    """Shared skeleton for LSTM and GRU."""

    def __init__(self, rnn_type: str, input_size: int = 1, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2, output_size: int = 1):
        super().__init__()
        self.rnn_type = rnn_type
        rnn_cls = nn.LSTM if rnn_type == "LSTM" else nn.GRU
        self.rnn = rnn_cls(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(nn.Linear(hidden_size, 32), nn.ReLU(), nn.Linear(32, output_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])


class LSTMModel(_RNNBase):
    def __init__(self, **kwargs):
        super().__init__(rnn_type="LSTM", **kwargs)


class GRUModel(_RNNBase):
    def __init__(self, **kwargs):
        super().__init__(rnn_type="GRU", **kwargs)


def train_model(model: nn.Module, train_scaled: np.ndarray, seq_len: int = 60,
                epochs: int = 30, batch_size: int = 32, lr: float = 1e-3) -> List[float]:
    model.to(DEVICE)
    X, y = make_sequences(train_scaled.flatten(), seq_len)
    dataset = TensorDataset(
        torch.tensor(X[..., np.newaxis] if X.ndim == 2 else X),
        torch.tensor(y).unsqueeze(1),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    losses = []

    model.train()
    for epoch in range(epochs):
        ep_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb)
            loss = crit(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
        losses.append(ep_loss / len(loader))
        if (epoch + 1) % 10 == 0:
            log.info(f"    Epoch {epoch+1}/{epochs}  loss={losses[-1]:.6f}")
    return losses

def predict_test_rnn(model: nn.Module, train_scaled: np.ndarray, test_scaled: np.ndarray,
                     scaler, seq_len: int = 60) -> np.ndarray:
    """Walk-forward 1-step prediction over the test set."""
    model.eval()
    all_data = np.concatenate([train_scaled.flatten(), test_scaled.flatten()])
    preds_sc = []

    with torch.no_grad():
        for i in range(len(test_scaled)):
            idx = len(train_scaled) + i
            seq = all_data[idx - seq_len: idx].reshape(1, seq_len, 1)
            x = torch.tensor(seq, dtype=torch.float32).to(DEVICE)
            p = model(x).cpu().numpy().flatten()[0]
            preds_sc.append(p)
            all_data[idx] = test_scaled.flatten()[i]

    return scaler.inverse_transform(np.array(preds_sc).reshape(-1, 1)).flatten()


def forecast_future_rnn(model: nn.Module, train_scaled: np.ndarray, test_scaled: np.ndarray,
                        scaler, seq_len: int = 60, n_periods: int = 5) -> np.ndarray:
    """Autoregressively forecast n_periods beyond test window."""
    model.eval()
    all_data = np.concatenate([train_scaled.flatten(), test_scaled.flatten()])
    seed = all_data[-seq_len:].copy()
    fc_sc = []

    with torch.no_grad():
        for _ in range(n_periods):
            x = torch.tensor(seed.reshape(1, seq_len, 1), dtype=torch.float32).to(DEVICE)
            p = model(x).cpu().numpy().flatten()[0]
            fc_sc.append(p)
            seed = np.roll(seed, -1)
            seed[-1] = p

    return scaler.inverse_transform(np.array(fc_sc).reshape(-1, 1)).flatten()

def _run_rnn_pipeline(processed: Dict[str, Dict], rnn_type: str, seq_len: int = 60,
                      epochs: int = 30, hidden_size: int = 64, n_forecast: int = 5,
                      save: bool = True) -> Tuple[Dict[str, pd.Series], Dict[str, np.ndarray], list]:
    test_preds: Dict[str, pd.Series] = {}
    future_fc: Dict[str, np.ndarray] = {}
    metrics = []
    model_label = rnn_type

    for ticker, d in processed.items():
        log.info(f"{model_label} | {ticker}")
        train, test = d["train"], d["test"]
        tr_sc, te_sc, scaler = d["train_scaled"], d["test_scaled"], d["scaler"]

        if len(tr_sc) < seq_len + 10:
            log.warning(f"Not enough data for {ticker}, skipping.")
            continue

        model_cls = LSTMModel if rnn_type == "LSTM" else GRUModel
        model = model_cls(hidden_size=hidden_size)
        train_model(model, tr_sc, seq_len=seq_len, epochs=epochs)

        raw_preds = predict_test_rnn(model, tr_sc, te_sc, scaler, seq_len)
        pred = pd.Series(raw_preds, index=test.index, name=f"{model_label}_Pred")
        fc = forecast_future_rnn(model, tr_sc, te_sc, scaler, seq_len, n_forecast)

        m_dict = evaluate(test.values, pred.values, model_label, ticker)
        test_preds[ticker] = pred
        future_fc[ticker] = fc
        metrics.append(m_dict)

        if save:
            out = pd.DataFrame({"Date": test.index, "Actual": test.values, f"{model_label}_Pred": pred.values})
            fp = FORECASTS_DIR / f"{ticker.replace('.', '_')}_{model_label.lower()}.csv"
            out.to_csv(fp, index=False)

    return test_preds, future_fc, metrics


def run_lstm_pipeline(processed, n_forecast=5, epochs=30, save=True):
    return _run_rnn_pipeline(processed, "LSTM", n_forecast=n_forecast, epochs=epochs, save=save)


def run_gru_pipeline(processed, n_forecast=5, epochs=30, save=True):
    return _run_rnn_pipeline(processed, "GRU", n_forecast=n_forecast, epochs=epochs, save=save)
