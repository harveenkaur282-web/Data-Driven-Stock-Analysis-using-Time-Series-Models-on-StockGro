"""
allocation.py
-------------
Task 5: Portfolio Allocation logic.
Outputs to deliverables/task5_portfolio/
"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np

log = logging.getLogger(__name__)

# Output Path
OUT_DIR = Path(__file__).resolve().parents[2] / "deliverables" / "task5_portfolio"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def build_portfolio(
    tickers: list,
    sector_map: dict,
    stock_names: dict,
    future_forecasts: dict,
    current_prices: dict,
    volatility_estimates: dict,
    corr_weights: pd.Series,
    sector_returns: pd.DataFrame,
    total_capital: float = 1_000_000,
    save: bool = True
) -> pd.DataFrame:
    """
    Combined scoring and allocation.
    - Forecast score (predicted return)
    - Volatility score (inverse vol)
    - Correlation score
    - Sector momentum score
    """
    results = []
    
    for ticker in tickers:
        # 1. Forecast Return (5-day predicted %)
        curr_p = current_prices.get(ticker)
        fc_p = future_forecasts.get(ticker)[-1] if ticker in future_forecasts else curr_p
        pred_ret = (fc_p - curr_p) / curr_p if curr_p else 0
        
        # 2. Vol Score
        vol = volatility_estimates.get(ticker, 0.2)
        vol_score = 1 / vol if vol > 0 else 1
        
        # 3. Corr Score
        corr_score = corr_weights.get(ticker, 1/len(tickers))
        
        # 4. Sector Score
        sector = sector_map.get(ticker, "Unknown")
        sec_ret = sector_returns.loc[ticker, '6M_CumReturn'] if ticker in sector_returns.index else 0
        
        # Combined score (weighted)
        score = (pred_ret * 0.4) + (vol_score * 0.2) + (corr_score * 0.2) + (sec_ret * 0.2)
        
        results.append({
            "Ticker": ticker,
            "Name": stock_names.get(ticker, ticker),
            "Sector": sector,
            "Price": curr_p,
            "Predicted_Return_5D": round(pred_ret, 4),
            "Score": score
        })
        
    df = pd.DataFrame(results)
    
    # Normalize score to weights
    df['Weight'] = df['Score'] / df['Score'].sum()
    df['Amount'] = df['Weight'] * total_capital
    df['Shares_Approx'] = (df['Amount'] / df['Price']).round(0)
    
    if save:
        df.to_csv(OUT_DIR / "portfolio_allocation.csv", index=False)
        log.info(f"Portfolio allocation saved to {OUT_DIR / 'portfolio_allocation.csv'}")
        
    return df
