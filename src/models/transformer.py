from __future__ import annotations
import logging
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.utils.metrics import evaluate
from src.models.lstm import make_sequences

log = logging.getLogger(__name__)

FORECASTS_DIR = Path(__file__).resolve().parents[2] / "results" / "forecasts"
FORECASTS_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class StockTransformer(nn.Module):
    def __init__(self, d_model: int = 64, nhead: int = 4, num_encoder_layers: int = 2,
                 dim_feedforward: int = 128, dropout: float = 0.1, seq_len: int = 60):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=seq_len + 1, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.fc_out = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        x = x[:, -1, :]
        return self.fc_out(x)


def train_transformer(model: StockTransformer, train_scaled: np.ndarray, seq_len: int = 60,
                      epochs: int = 30, batch_size: int = 32, lr: float = 5e-4) -> list:
    model.to(DEVICE)
    X, y = make_sequences(train_scaled.flatten(), seq_len)
    X_t = torch.tensor(X[..., np.newaxis] if X.ndim == 2 else X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
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
        sched.step()
        losses.append(ep_loss / len(loader))
        if (epoch + 1) % 10 == 0:
            log.info(f"    Epoch {epoch+1}/{epochs}  loss={losses[-1]:.6f}")
    return losses


def predict_test_transformer(model, train_scaled, test_scaled, scaler, seq_len=60):
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
    return scaler.inverse_transform(np.array(preds_sc).reshape(-1, 1)).flatten()


def forecast_future_transformer(model, train_scaled, test_scaled, scaler,
                                seq_len=60, n_periods=5):
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


def run_transformer_pipeline(processed: Dict[str, Dict], seq_len: int = 60,
                             epochs: int = 30, n_forecast: int = 5,
                             save: bool = True):
    test_preds, future_fc, metrics = {}, {}, []

    for ticker, d in processed.items():
        log.info(f"Transformer | {ticker}")
        train, test = d["train"], d["test"]
        tr_sc, te_sc, scaler = d["train_scaled"], d["test_scaled"], d["scaler"]

        if len(tr_sc) < seq_len + 10:
            log.warning(f"Not enough data for {ticker}, skipping.")
            continue

        model = StockTransformer(seq_len=seq_len)
        train_transformer(model, tr_sc, seq_len=seq_len, epochs=epochs)

        raw_preds = predict_test_transformer(model, tr_sc, te_sc, scaler, seq_len)
        pred = pd.Series(raw_preds, index=test.index, name="Transformer_Pred")
        fc = forecast_future_transformer(model, tr_sc, te_sc, scaler, seq_len, n_forecast)

        m_dict = evaluate(test.values, pred.values, "Transformer", ticker)
        test_preds[ticker] = pred
        future_fc[ticker] = fc
        metrics.append(m_dict)

        if save:
            out = pd.DataFrame({"Date": test.index, "Actual": test.values, "Transformer_Pred": pred.values})
            fp = FORECASTS_DIR / f"{ticker.replace('.', '_')}_transformer.csv"
            out.to_csv(fp, index=False)

    return test_preds, future_fc, metrics
