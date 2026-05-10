
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

STOCK_UNIVERSE: Dict[str, str] = {
    # Banking
    "HDFCBANK.NS": "HDFC Bank",
    "ICICIBANK.NS": "ICICI Bank",
    # IT
    "TCS.NS": "TCS",
    "INFY.NS": "Infosys",
    # Pharma
    "SUNPHARMA.NS": "Sun Pharma",
    "DRREDDY.NS": "Dr. Reddy's",
    # FMCG
    "HINDUNILVR.NS": "Hindustan Unilever",
    "ITC.NS": "ITC",
    # Auto
    "MARUTI.NS": "Maruti Suzuki"
}

SECTOR_MAP: Dict[str, str] = {
    "HDFCBANK.NS": "Banking",
    "ICICIBANK.NS": "Banking",
    "TCS.NS": "IT",
    "INFY.NS": "IT",
    "SUNPHARMA.NS": "Pharma",
    "DRREDDY.NS": "Pharma",
    "HINDUNILVR.NS": "FMCG",
    "ITC.NS": "FMCG",
    "MARUTI.NS": "Auto",
    "TATAMOTORS.NS": "Auto",
}

START_DATE = "2021-01-01"
END_DATE   = "2025-12-31"
INTERVAL   = "1d"

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def fetch_single(
    ticker: str,
    start: str = START_DATE,
    end: str = END_DATE,
    interval: str = INTERVAL,
) -> pd.DataFrame:
    """Download OHLCV data for one ticker and return a clean DataFrame."""
    log.info(f"Downloading {ticker} …")
    df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=True, progress=False)
    if df.empty:
        log.warning(f"No data returned for {ticker}.")
        return df
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df["Ticker"] = ticker
    return df


def fetch_all(
    universe: Dict[str, str] = STOCK_UNIVERSE,
    start: str = START_DATE,
    end: str = END_DATE,
    save: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Fetch all tickers and optionally persist to data/raw/<TICKER>.csv."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    data: Dict[str, pd.DataFrame] = {}

    for ticker in universe:
        df = fetch_single(ticker, start=start, end=end)
        if df.empty:
            continue
        data[ticker] = df
        if save:
            path = RAW_DIR / f"{ticker.replace('.', '_')}.csv"
            df.to_csv(path)
            log.info(f"  Saved → {path}")

    log.info(f"Fetched {len(data)}/{len(universe)} tickers successfully.")
    return data


def load_raw(ticker: str) -> pd.DataFrame:
    """Load a previously saved raw CSV for a ticker."""
    path = RAW_DIR / f"{ticker.replace('.', '_')}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Raw file not found: {path}. Run fetch_all() first.")
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    return df


def load_all_raw(universe: Dict[str, str] = STOCK_UNIVERSE) -> Dict[str, pd.DataFrame]:
    """Load all raw CSVs into a dict."""
    return {ticker: load_raw(ticker) for ticker in universe}


if __name__ == "__main__":
    data = fetch_all()
    for t, df in data.items():
        print(f"{t}: {len(df)} rows, {df.index[0].date()} → {df.index[-1].date()}")