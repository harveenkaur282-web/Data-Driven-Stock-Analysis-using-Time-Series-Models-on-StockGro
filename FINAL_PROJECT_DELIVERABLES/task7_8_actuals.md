## Task 7: StockGro Execution Summary (Day 1-2)

- Invested Amount (deployed): ₹997,778.36
- Transaction Charges: ₹-997.78
- Unrealised P/L at Day 2 close: ₹928.47
- Final Mark-to-Market Portfolio Value: ₹997,709.05
- Net P/L (incl. txn charges): ₹-69.31

### Holdings at Day 2 (actuals)

| Ticker | Qty | Avg Price (₹) | Market Price (₹) | Purchase Val (₹) | Market Val (₹) | P/L (₹) | P/L (%) | Alloc Amount (₹) | Diff (Market - Alloc) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HDFCBANK.NS | 127 | 745.66 | 759.28 | 94699.09 | 96428.56 | 1729.47 | 1.83% | 94922.29 | 1506.27 |
| ICICIBANK.NS | 51 | 1220.25 | 1235.34 | 62232.75 | 63002.34 | 769.59 | 1.24% | 61907.73 | 1094.61 |
| TCS.NS | 25 | 2252.32 | 2204.19 | 56308.00 | 55104.75 | -1203.25 | -2.14% | 57777.64 | -2672.89 |
| INFY.NS | 100 | 1126.63 | 1085.41 | 112663.00 | 108541.00 | -4122.00 | -3.66% | 113015.78 | -4474.78 |
| SUNPHARMA.NS | 61 | 1829.84 | 1845.28 | 111620.24 | 112562.08 | 941.84 | 0.84% | 111357.65 | 1204.43 |
| DRREDDY.NS | 74 | 1251.01 | 1295.70 | 92574.74 | 95881.80 | 3307.06 | 3.57% | 92417.38 | 3464.42 |
| HINDUNILVR.NS | 45 | 2227.94 | 2227.14 | 100257.30 | 100221.30 | -36.00 | -0.04% | 99723.76 | 497.54 |
| ITC.NS | 276 | 302.20 | 301.23 | 83407.20 | 83139.48 | -267.72 | -0.32% | 83371.44 | -231.96 |
| MARUTI.NS | 22 | 12909.82 | 12901.16 | 284016.04 | 283825.52 | -190.52 | -0.07% | 285506.31 | -1680.79 |

### Portfolio Performance Summary

- Total Allocated (per allocation file): ₹1,000,000.00
- Sum Purchase Value (from Day1 trades): ₹997,778.36
- Sum Market Value (Day2 close): ₹998,706.83
- Aggregate P/L: ₹928.47

### Task 8: Predicted vs Actual Outcomes

For each stock, the model forecasts (used to construct the portfolio) should be compared to the actual Day 1/Day 2 prices. Below are the recorded actuals and portfolio-level outcomes. Detailed per-model forecast vs actual comparisons are in `results/forecasts/` for each model and ticker.

> Note: Directional accuracy and MAPE for the live 2-day window should be computed using the model forecasts saved under `results/forecasts/` and the Day1/Day2 actual prices. Those computations can be added if you provide the model forecast values for the two traded days.