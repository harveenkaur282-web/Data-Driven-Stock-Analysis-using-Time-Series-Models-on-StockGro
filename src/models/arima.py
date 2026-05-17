#uses pmdarima.auto_arima for AIC/BIC-guided order selection
#validates residuals (Ljung-Box)
#Produces test-period predicitons and n-step ahead forecasts
from __future__ import annotations
import logging
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import pmdarima as pm
from statsmodels.stats.diagnostic import acorr_ljungbox

from src.utils.metrics import evaluate

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

FORECASTS_DIR = Path(__file__).resolve().parents[2] / "results" / "forecasts"
FORECASTS_DIR.mkdir(parents=True, exist_ok=True)


def fit_arima(
    train: pd.Series,
    seasonal: bool = False,
    m: int = 5,
    information_criterion: str = "aic",
    max_p: int = 5,
    max_q: int = 5,
) -> pm.arima.ARIMA:
    log.info(f"  Fitting auto_arima (seasonal={seasonal}, m={m}) …")
    model = pm.auto_arima(
        train,
        seasonal=seasonal,
        m=m,
        information_criterion=information_criterion,
        max_p=max_p, max_q=max_q,
        max_P=2, max_Q=2,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
    )
    log.info(f"  Best order: {model.order}  seasonal_order: {model.seasonal_order}")
    return model

def residual_diagnostics(model: pm.arima.ARIMA) -> Dict:
    resid = pd.Series(model.resid())
    lb = acorr_ljungbox(resid, lags=[10, 20], return_df=True)
    return {
        "lb_stat_10":  round(float(lb["lb_stat"].iloc[0]), 4),
        "lb_pval_10":  round(float(lb["lb_pvalue"].iloc[0]), 4),
        "lb_stat_20":  round(float(lb["lb_stat"].iloc[1]), 4),
        "lb_pval_20":  round(float(lb["lb_pvalue"].iloc[1]), 4),
        "white_noise": bool(lb["lb_pvalue"].min() > 0.05),
    }

def predict_test(
    model: pm.arima.ARIMA,
    test: pd.Series,
) -> pd.Series:
    predictions = []
    # Work on a copy to avoid side effects if reused
    current_model = model

    for i in range(len(test)):
        try:
            fc = current_model.predict(n_periods=1)
            # Robust extraction: handle both Series and Numpy array
            p_val = fc.iloc[0] if hasattr(fc, 'iloc') else fc[0]
            predictions.append(float(p_val))
            # Update with the next observation (as a Series to preserve index/metadata)
            current_model.update(test.iloc[i:i+1])
        except Exception as e:
            log.error(f"  Error at step {i}: {e}")
            predictions.append(np.nan)

    return pd.Series(predictions, index=test.index, name="ARIMA_Pred")


def forecast_future(
    model: pm.arima.ARIMA,
    train: pd.Series,
    test: pd.Series,
    n_periods: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Re-fit on train+test, forecast n_periods ahead. Returns (fc, lower, upper)."""
    full = pd.concat([train, test])
    # Refit auto_arima on full data for future forecast
    m = pm.auto_arima(
        full, seasonal=False, stepwise=True,
        suppress_warnings=True, error_action="ignore",
    )
    fc, conf = m.predict(n_periods=n_periods, return_conf_int=True)
    return fc, conf[:, 0], conf[:, 1]
#this function can be called then in notebooks to get the forecast on next days (outside of Jan 2021-dec 2025)

def run_arima_pipeline(
    processed: Dict[str, Dict],
    n_forecast: int = 5,
    save: bool = True,
) -> Tuple[Dict[str, pd.Series], Dict[str, np.ndarray], list]:
    """Run ARIMA for all tickers."""
    import traceback
    test_preds: Dict[str, pd.Series] = {}
    future_fc: Dict[str, np.ndarray] = {}
    metrics = []

    for ticker, d in processed.items():
        log.info(f"ARIMA | {ticker}")
        train, test = d["train"], d["test"]
        try:
            model = fit_arima(train)
            pred = predict_test(model, test)
            fc, lo, hi = forecast_future(model, train, test, n_periods=n_forecast)
            lb_pval = acorr_ljungbox(model.resid(), lags=[10], return_df=True)['lb_pvalue'].iloc[0]
            
            m_dict = evaluate(test.values, pred.values, "ARIMA", ticker)
            m_dict["WhiteNoise"] = bool(lb_pval > 0.05)
        except Exception as e:
            log.error(f"ARIMA failed for {ticker}: {e}")
            log.error(traceback.format_exc())
            pred = pd.Series(np.nan, index=test.index)
            fc = np.full(n_forecast, np.nan)
            m_dict = evaluate(test.values, np.full(len(test), np.nan), "ARIMA", ticker)

        test_preds[ticker] = pred
        future_fc[ticker] = fc
        metrics.append(m_dict)

        if save:
            out = pd.DataFrame({"Date": test.index, "Actual": test.values, "ARIMA_Pred": pred.values})
            fp = FORECASTS_DIR / f"{ticker.replace('.', '_')}_arima.csv"
            out.to_csv(fp, index=False)

    return test_preds, future_fc, metrics
