"""
config.py — Single source of truth for project-wide parameters.
"""

# ── Date ranges ────────────────────────────────────────────────────────────────
TRAIN_START = "2021-01-01"
TRAIN_END   = "2025-06-30"

TEST_START  = "2025-07-01"
TEST_END    = "2025-12-31"

# ── Stock universe (9 stocks, 5 sectors) ──────────────────────────────────────
STOCK_UNIVERSE = {
    "HDFCBANK.NS":   "HDFC Bank",
    "ICICIBANK.NS":  "ICICI Bank",
    "TCS.NS":        "TCS",
    "INFY.NS":       "Infosys",
    "SUNPHARMA.NS":  "Sun Pharma",
    "DRREDDY.NS":    "Dr. Reddy's",
    "HINDUNILVR.NS": "Hindustan Unilever",
    "ITC.NS":        "ITC",
    "MARUTI.NS":     "Maruti Suzuki",
}

SECTOR_MAP = {
    "HDFCBANK.NS":   "Banking",
    "ICICIBANK.NS":  "Banking",
    "TCS.NS":        "IT",
    "INFY.NS":       "IT",
    "SUNPHARMA.NS":  "Pharma",
    "DRREDDY.NS":    "Pharma",
    "HINDUNILVR.NS": "FMCG",
    "ITC.NS":        "FMCG",
    "MARUTI.NS":     "Auto",
}

# ── Forecasting ───────────────────────────────────────────────────────────────
FORECAST_HORIZON = 5        # days ahead to forecast
SEQ_LEN          = 60       # lookback window for LSTM / GRU / Transformer
TOTAL_CAPITAL    = 1_000_000  # ₹10,00,000