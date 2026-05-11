"""
volatility.py
-------------
Task 4 – Volatility Estimation:
- GARCH(1,1)
- Rolling Volatility
- Output to deliverables/task4_analysis/
"""

import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
from arch import arch_model

log = logging.getLogger(__name__)

# Output Path
OUT_DIR = Path(__file__).resolve().parents[2] / "deliverables" / "task4_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def log_returns(prices: pd.Series) -> pd.Series:
    return (np.log(prices / prices.shift(1)).dropna() * 100).rename("LogReturn_pct")

def rolling_vol(ret: pd.Series, window: int = 30) -> pd.Series:
    return ret.rolling(window=window).std() * np.sqrt(252)

def fit_garch(ret: pd.Series) -> Tuple[float, object]:
    """Fit GARCH(1,1) and return forecast."""
    try:
        model = arch_model(ret, vol='Garch', p=1, q=1, rescale=False)
        res = model.fit(disp='off')
        forecast = res.forecast(horizon=1)
        # Annualized forecast
        fc_vol = np.sqrt(forecast.variance.values[-1, 0] * 252)
        return fc_vol, res
    except Exception as e:
        log.warning(f"GARCH fit failed: {e}")
        return np.nan, None

def volatility_summary(raw_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Computes volatility metrics for all stocks."""
    results = []
    for ticker, df in raw_data.items():
        prices = df['Close'].squeeze()
        rets = log_returns(prices)
        
        rv = rolling_vol(rets).iloc[-1]
        gv, _ = fit_garch(rets)
        
        results.append({
            "Ticker": ticker,
            "Rolling_Vol_Ann": round(rv, 4),
            "GARCH_FcstVol": round(gv, 4) if not np.isnan(gv) else None,
            "Sharpe_Approx": round(float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0, 4)
        })
    
    vol_df = pd.DataFrame(results).set_index("Ticker")
    
    # Save Deliverable
    vol_df.to_csv(OUT_DIR / "volatility_metrics.csv")
    log.info(f"Volatility summary saved to {OUT_DIR / 'volatility_metrics.csv'}")
    
    return vol_df
