from __future__ import annotations
from typing import Dict

import numpy as np

def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    if mask.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((actual[mask] - predicted[mask]) ** 2)))

def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = ~(np.isnan(actual) | np.isnan(predicted)) & (actual != 0)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)

def directional_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
#Percentage of times the model correctly predicts the direction of change.
#Compares sign(Δactual) vs sign(Δpredicted) for consecutive steps.
    if len(actual) < 2 or len(predicted) < 2:
        return float("nan")
    actual_dir = np.sign(np.diff(actual))
    pred_dir   = np.sign(np.diff(predicted))
    mask = ~(np.isnan(actual_dir) | np.isnan(pred_dir))
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(actual_dir[mask] == pred_dir[mask]) * 100)

def evaluate(
    actual: np.ndarray,
    predicted: np.ndarray,
    model_name: str = "",
    ticker: str = "",
) -> Dict:
    return {
        "Ticker":  ticker,
        "Model":   model_name,
        "RMSE":    round(rmse(actual, predicted), 4),
        "MAPE":    round(mape(actual, predicted), 4),
        "DirAcc":  round(directional_accuracy(actual, predicted), 2),
    }
