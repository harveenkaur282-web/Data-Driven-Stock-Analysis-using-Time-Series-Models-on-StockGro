# stock selection using rolling std dev,
# seasonal decomposition, and sector-based rationale.

from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

log = logging.getLogger(__name__)

FIGURES_DIR = Path(__file__).resolve().parents[2] / "deliverables" / "images"
TABLES_DIR  = Path(__file__).resolve().parents[2] / "deliverables" / "tables"

def rolling_volatility(prices: pd.Series, window: int = 30) -> pd.Series:
    """Annualised rolling std-dev of log returns."""
    log_ret = np.log(prices / prices.shift(1)).dropna()
    return log_ret.rolling(window).std() * np.sqrt(252)

def volatility_summary(data: Dict[str, pd.DataFrame], window: int = 30) -> pd.DataFrame:
#Returns a DataFrame with mean / max / recent annualised vol for each ticker.
#Used to justify stock selection with 'interesting volatility profiles. 
    rows = []
    for ticker, df in data.items():
        price = df["Close"].squeeze()
        rv = rolling_volatility(price, window).dropna()
        rows.append({
            "Ticker": ticker,
            "Mean_AnnVol": round(rv.mean(), 4),
            "Max_AnnVol":  round(rv.max(),  4),
            "Recent_AnnVol": round(rv.iloc[-1], 4),
        })
    return pd.DataFrame(rows).set_index("Ticker")

def decompose_stock(
    prices: pd.Series,
    period: int = 252,   # annual trading days
    model: str = "additive",
) -> object:
#Doubt: is this STL STYLE OR SOMETHING DIFFERENT?
    prices = prices.dropna()
    result = seasonal_decompose(prices, model=model, period=period, extrapolate_trend="freq")
    return result

def trend_strength(decomp_result) -> float:
#This function measures the trend strength as the ratio of trend variance to 
# (trend + residual) variance. Range 0-1; closer to 1 means strong trend.
    trend   = decomp_result.trend.dropna()
    resid   = decomp_result.resid.dropna()
    aligned = trend.align(resid, join="inner")
    t_var   = np.var(aligned[0])
    r_var   = np.var(aligned[1])
    if t_var + r_var == 0:
        return 0.0
    return round(float(t_var / (t_var + r_var)), 4)


def decomposition_summary(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
#trend strengh for each ticker
    rows = []
    for ticker, df in data.items():
        price = df["Close"].squeeze()
        try:
            dec = decompose_stock(price)
            ts  = trend_strength(dec)
            direction = "upward" if dec.trend.dropna().iloc[-1] > dec.trend.dropna().iloc[0] else "downward"
        except Exception as e:
            log.warning(f"Decompose failed for {ticker}: {e}")
            ts, direction = float("nan"), "unknown"
        rows.append({"Ticker": ticker, "Trend_Strength": ts, "Trend_Direction": direction})
    return pd.DataFrame(rows).set_index("Ticker")

def sector_momentum(
    data: Dict[str, pd.DataFrame],
    sector_map: Dict[str, str],
    lookback: int = 126,  # ~6 months
) -> pd.DataFrame:
#computes 6-month cumulative return per stock and rank by sector.
    rows = []
    for ticker, df in data.items():
        price = df["Close"].squeeze().dropna()
        if len(price) < lookback:
            cum_ret = float("nan")
        else:
            cum_ret = round(float(price.iloc[-1] / price.iloc[-lookback] - 1), 4)
        rows.append({
            "Ticker": ticker,
            "Sector": sector_map.get(ticker, "Unknown"),
            "6M_CumReturn": cum_ret,
        })
    df_out = pd.DataFrame(rows).set_index("Ticker").sort_values("6M_CumReturn", ascending=False)
    return df_out

def screening_report(
    data: Dict[str, pd.DataFrame],
    sector_map: Dict[str, str],
    stock_names: Dict[str, str],
) -> pd.DataFrame:

    vol_df  = volatility_summary(data)
    dec_df  = decomposition_summary(data)
    mom_df  = sector_momentum(data, sector_map)

    report = vol_df.join(dec_df).join(mom_df)
    report["Name"] = report.index.map(stock_names)
    report["Sector"] = report.index.map(sector_map)
    report = report.sort_values("6M_CumReturn", ascending=False)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out = TABLES_DIR / "stock_screening_report.csv"
    report.to_csv(out)
    log.info(f"Screening report saved → {out}")
    return report


if __name__ == "__main__":
    from fetch_data import load_all_raw, STOCK_UNIVERSE, SECTOR_MAP
    data = load_all_raw()
    rpt  = screening_report(data, SECTOR_MAP, STOCK_UNIVERSE)
    print(rpt.to_string())