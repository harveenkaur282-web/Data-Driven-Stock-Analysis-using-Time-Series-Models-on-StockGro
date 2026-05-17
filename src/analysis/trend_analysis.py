import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
import pandas as pd
from statsmodels.tsa.seasonal import STL

log = logging.getLogger(__name__)

# Output Path
OUT_DIR = Path(__file__).resolve().parents[2] / "deliverables" / "task4_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def calculate_trend_strength(res) -> float:
#Trend strength = max(0,1 - Var(resid)/Var(trend+resid)). Range 0-1; closer to 1 means strong trend.
    v_resid = res.resid.var()
    v_trend_resid = (res.trend + res.resid).var()
    return max(0, 1 - v_resid / v_trend_resid)

def trend_summary(raw_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    results = []
    for ticker, df in raw_data.items():
        prices = df['Close'].squeeze().dropna()      
        # STL Decomposition (period 252 for annual)
        stl = STL(prices, period=252, robust=True)
        res = stl.fit()       
        strength = calculate_trend_strength(res)
        
        # Direction based on last 20 days
        recent_trend = res.trend.iloc[-20:]
        direction = "Upward" if recent_trend.iloc[-1] > recent_trend.iloc[0] else "Downward"
        
        results.append({
            "Ticker": ticker,
            "Trend_Strength": round(strength, 4),
            "Trend_Direction": direction
        })
    
    trend_df = pd.DataFrame(results).set_index("Ticker")

    trend_df.to_csv(OUT_DIR / "trend_metrics.csv")
    log.info(f"Trend summary saved to {OUT_DIR / 'trend_metrics.csv'}")
    
    return trend_df
