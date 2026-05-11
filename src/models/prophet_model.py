"""
prophet_model.py
----------------
Task 3: Facebook Prophet forecasting.
"""

from __future__ import annotations
import logging
import warnings
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from prophet import Prophet

from src.utils.metrics import evaluate

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

FORECASTS_DIR = Path(__file__).resolve().parents[2] / "results" / "forecasts"
FORECASTS_DIR.mkdir(parents=True, exist_ok=True)


def series_to_prophet_df(series: pd.Series) -> pd.DataFrame:
    """Convert a DatetimeIndex price Series to Prophet's ds/y format."""
    df = series.reset_index()
    df.columns = ["ds", "y"]
    df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)
    return df


def fit_prophet(
    train: pd.Series,
    daily_seasonality: bool = True,
    weekly_seasonality: bool = True,
    yearly_seasonality: bool = True,
    changepoint_prior_scale: float = 0.05,
) -> Prophet:
    """Fit a Prophet model on the training series."""
    model = Prophet(
        daily_seasonality=daily_seasonality,
        weekly_seasonality=weekly_seasonality,
        yearly_seasonality=yearly_seasonality,
        changepoint_prior_scale=changepoint_prior_scale,
        interval_width=0.95,
    )
    train_df = series_to_prophet_df(train)
    model.fit(train_df)
    return model


def predict_test_prophet(model: Prophet, test: pd.Series) -> pd.Series:
    """Predict on test period dates."""
    future_df = pd.DataFrame({"ds": pd.to_datetime(test.index).tz_localize(None)})
    forecast = model.predict(future_df)
    pred = pd.Series(forecast["yhat"].values, index=test.index, name="Prophet_Pred")
    return pred


def forecast_future_prophet(
    model: Prophet,
    train: pd.Series,
    test: pd.Series,
    n_periods: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Re-fit on combined train+test, then forecast n business days ahead."""
    full = pd.concat([train, test])
    m2 = Prophet(
        daily_seasonality=True, weekly_seasonality=True,
        yearly_seasonality=True, changepoint_prior_scale=0.05,
        interval_width=0.95,
    )
    m2.fit(series_to_prophet_df(full))
    future = m2.make_future_dataframe(periods=n_periods, freq="B")
    fc = m2.predict(future).tail(n_periods)
    return fc["yhat"].values, fc["yhat_lower"].values, fc["yhat_upper"].values


def run_prophet_pipeline(
    processed: Dict[str, Dict],
    n_forecast: int = 5,
    save: bool = True,
) -> Tuple[Dict[str, pd.Series], Dict[str, np.ndarray], list]:
    """Run Prophet for all tickers."""
    test_preds: Dict[str, pd.Series] = {}
    future_fc: Dict[str, np.ndarray] = {}
    metrics = []

    for ticker, d in processed.items():
        log.info(f"Prophet | {ticker}")
        train, test = d["train"], d["test"]
        try:
            model = fit_prophet(train)
            pred = predict_test_prophet(model, test)
            fc, lo, hi = forecast_future_prophet(model, train, test, n_periods=n_forecast)
            m_dict = evaluate(test.values, pred.values, "Prophet", ticker)
        except Exception as e:
            log.error(f"Prophet failed for {ticker}: {e}")
            pred = pd.Series(np.nan, index=test.index)
            fc = np.full(n_forecast, np.nan)
            m_dict = evaluate(test.values, np.full(len(test), np.nan), "Prophet", ticker)

        test_preds[ticker] = pred
        future_fc[ticker] = fc
        metrics.append(m_dict)

        if save:
            out = pd.DataFrame({"Date": test.index, "Actual": test.values, "Prophet_Pred": pred.values})
            fp = FORECASTS_DIR / f"{ticker.replace('.', '_')}_prophet.csv"
            out.to_csv(fp, index=False)

    return test_preds, future_fc, metrics
