import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np

log = logging.getLogger(__name__)

OUT_DIR = Path(__file__).resolve().parents[2] / "deliverables" / "task4_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def return_correlation_matrix(raw_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Computes correlation matrix of returns."""
    all_rets = {}
    for ticker, df in raw_data.items():
        prices = df['Close'].squeeze()
        rets = prices.pct_change().dropna()
        all_rets[ticker] = rets
    
    ret_df = pd.DataFrame(all_rets)
    corr_matrix = ret_df.corr()

    corr_matrix.to_csv(OUT_DIR / "correlation_matrix.csv")
    log.info(f"Correlation matrix saved to {OUT_DIR / 'correlation_matrix.csv'}")
    
    return corr_matrix

def correlation_weights(corr_matrix: pd.DataFrame) -> pd.Series:
#simple inverse correlation weighting for diversification 
#doubt:? AND MAYBE FOR LESS RISK???
    """Simple inverse correlation weighting for diversification."""
    avg_corr = corr_matrix.mean()
    inv_corr = 1 - avg_corr
    weights = inv_corr / inv_corr.sum()
    return weights
