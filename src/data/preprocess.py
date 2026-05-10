from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller

log = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "preprocessed"
TRAIN_END  = "2025-06-30"
TEST_START = "2025-07-01"
TEST_END   = "2025-12-31"

def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill then backward-fill; drop any remaining NaN rows."""
    df = df.ffill().bfill()
    n_dropped = df.isna().any(axis=1).sum()
    if n_dropped:
        log.warning(f"Dropping {n_dropped} rows with remaining NaN after fill.")
        df = df.dropna()
    return df

def adf_test(series: pd.Series, signif: float = 0.05) -> Dict:
    result = adfuller(series.dropna(), autolag="AIC")
    return {
        "adf_stat":     round(result[0], 4),
        "p_value":      round(result[1], 6),
        "n_lags":       result[2],
        "n_obs":        result[3],
        "is_stationary": result[1] < signif,
    }


def make_stationary(series: pd.Series, max_diff: int = 2) -> Tuple[pd.Series, int]:
    d = 0
    s = series.copy()
    while d <= max_diff:
        res = adf_test(s)
        if res["is_stationary"]:
            break
        s = s.diff().dropna()
        d += 1
    if d > max_diff:
        log.warning("Series still non-stationary after max differencing.")
    return s, d


def stationarity_report(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Run ADF on Close prices and report d-order needed for each ticker."""
    rows = []
    for ticker, df in data.items():
        price = df["Close"].squeeze().dropna()
        adf0  = adf_test(price)
        _, d  = make_stationary(price)
        rows.append({
            "Ticker":        ticker,
            "ADF_Stat":      adf0["adf_stat"],
            "P_Value":       adf0["p_value"],
            "IsStationary":  adf0["is_stationary"],
            "D_Order":       d,
        })
    return pd.DataFrame(rows).set_index("Ticker")


def compute_log_returns(prices: pd.Series) -> pd.Series:
    return np.log(prices / prices.shift(1)).dropna().rename("LogReturn")


def train_test_split_ts(
    series: pd.Series,
    train_end: str = TRAIN_END,
    test_start: str = TEST_START,
    test_end:   str = TEST_END,
) -> Tuple[pd.Series, pd.Series]:
    train = series.loc[:train_end]
    test  = series.loc[test_start:test_end]
    return train, test


def scale_series(
    train: pd.Series,
    test:  pd.Series,
    feature_range: Tuple[float, float] = (0, 1),
) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    scaler = MinMaxScaler(feature_range=feature_range)
    train_scaled = scaler.fit_transform(train.values.reshape(-1, 1))
    test_scaled  = scaler.transform(test.values.reshape(-1, 1))
    return train_scaled, test_scaled, scaler

def preprocess_all(
    raw_data: Dict[str, pd.DataFrame],
    save: bool = True,
) -> Dict[str, Dict]:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out: Dict[str, Dict] = {}

    for ticker, df in raw_data.items():
        log.info(f"Preprocessing {ticker} …")
        df_clean = handle_missing(df.copy())
        price    = df_clean["Close"].squeeze().sort_index()
        _, d_order = make_stationary(price)
        log_ret = compute_log_returns(price)
        train, test = train_test_split_ts(price)
        tr_sc, te_sc, scaler = scale_series(train, test)

        out[ticker] = {
            "close":        price,
            "log_returns":  log_ret,
            "train":        train,
            "test":         test,
            "train_scaled": tr_sc,
            "test_scaled":  te_sc,
            "scaler":       scaler,
            "d_order":      d_order,
            "df_clean":     df_clean,
        }

        if save:
            fp = PROCESSED_DIR / f"{ticker.replace('.', '_')}_processed.csv"
            df_clean.to_csv(fp)

    log.info("Preprocessing complete.")
    return out


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, str(Path(__file__).parents[2]))
    from src.data.fetch_data import load_all_raw
    raw = load_all_raw()
    pp  = preprocess_all(raw)
    for t, d in pp.items():
        print(f"{t}: train={len(d['train'])}  test={len(d['test'])}  d={d['d_order']}")