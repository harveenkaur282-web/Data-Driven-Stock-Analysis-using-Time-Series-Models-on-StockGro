import glob
import pandas as pd
import numpy as np
from pathlib import Path

FORECAST_DIR = Path('results/forecasts')
ACTUALS_FILE = Path('deliverables/task7_stockgro/actuals.csv')
MODELS = ['ARIMA', 'Prophet', 'LSTM', 'GRU', 'Transformer']

def find_pred_column(df, model_name):
    """Find the forecast column in the dataframe."""
    model_lower = model_name.lower()
    for col in df.columns:
        if model_lower in col.lower():
            return col
    # Fallback: look for any *_Pred column
    for col in df.columns:
        if 'pred' in col.lower():
            return col
    return None

def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))

def mape(a, b):
    mask = a != 0
    if not mask.any():
        return np.nan
    return np.mean(np.abs((a[mask] - b[mask]) / a[mask])) * 100

def directional_accuracy(actual_values):
    """
    For 2-day window: check if direction (up/down) is predicted correctly.
    actual_values: [day1_actual, day2_actual]
    Returns: 100 if direction matches (day2 > day1 means up), else 0
    """
    if len(actual_values) != 2:
        return np.nan
    if actual_values[0] == actual_values[1]:
        return np.nan
    return 100 if (actual_values[1] > actual_values[0]) else 0

def main():
    # Load actuals
    actuals_df = pd.read_csv(ACTUALS_FILE)
    
    # Map ticker to day1 (Avg_Price) and day2 (Market_Price)
    actuals_map = {}
    for _, row in actuals_df.iterrows():
        ticker = row['Ticker']
        actuals_map[ticker] = {
            'day1': float(row['Avg_Price']),
            'day2': float(row['Market_Price'])
        }
    
    print("Extracted Actuals:")
    for t, vals in actuals_map.items():
        print(f"  {t}: Day1={vals['day1']:.2f}, Day2={vals['day2']:.2f}")
    
    # For each model, compute 2-day metrics
    results = {}
    
    for model in MODELS:
        results[model] = {
            'mape_values': [],
            'dirac_values': [],
            'per_ticker': {}
        }
        
        # Find all forecast files for this model (use lowercase for filenames)
        pattern = str(FORECAST_DIR / f'*_{model.lower()}.csv')
        files = sorted(glob.glob(pattern))
        print(f"\n{model}: Found {len(files)} forecast files")
        
        for fpath in files:
            fname = Path(fpath).name
            ticker_part = Path(fpath).stem.replace(f'_{model.lower()}', '').replace('_', '.')
            
            if ticker_part not in actuals_map:
                continue
            
            try:
                df = pd.read_csv(fpath, parse_dates=['Date'])
                pred_col = find_pred_column(df, model)
                
                if pred_col is None:
                    print(f"    WARNING: No prediction column found in {Path(fpath).name}, cols={list(df.columns)}")
                    continue
                
                # Get last 2 rows (Day 1 and Day 2 predictions)
                if len(df) < 2:
                    print(f"  WARNING: {fpath} has < 2 rows")
                    continue
                
                pred_day1 = float(df.iloc[-2][pred_col])
                pred_day2 = float(df.iloc[-1][pred_col])
                
                actual_day1 = actuals_map[ticker_part]['day1']
                actual_day2 = actuals_map[ticker_part]['day2']
                
                # Compute metrics
                m = mape(np.array([actual_day1, actual_day2]), 
                         np.array([pred_day1, pred_day2]))
                # Directional accuracy: both days' direction must match
                pred_dir = 1 if pred_day2 > pred_day1 else -1
                actual_dir = 1 if actual_day2 > actual_day1 else -1
                dirac = 100 if (pred_dir == actual_dir) else 0
                
                results[model]['per_ticker'][ticker_part] = {
                    'mape': m,
                    'dirac': dirac,
                    'actual': [actual_day1, actual_day2],
                    'pred': [pred_day1, pred_day2]
                }
                
                results[model]['mape_values'].append(m)
                results[model]['dirac_values'].append(dirac)
                
                print(f"    {ticker_part}: MAPE={m:.2f}%, DirAcc={dirac:.0f}%")
            except Exception as e:
                print(f"  ERROR processing {fpath}: {e}")
    
    # Print summary by model
    print("\n\n=== SUMMARY BY MODEL ===\n")
    summary_data = []
    for model in MODELS:
        if results[model]['mape_values']:
            avg_mape = np.nanmean(results[model]['mape_values'])
            avg_dirac = np.nanmean(results[model]['dirac_values'])
            count = len(results[model]['mape_values'])
            summary_data.append({
                'Model': model,
                'Avg_MAPE_%': round(avg_mape, 2),
                'Avg_DirAcc_%': round(avg_dirac, 2),
                'Tickers': count
            })
            print(f"{model}: Avg MAPE={avg_mape:.2f}%, Avg DirAcc={avg_dirac:.2f}% ({count} tickers)")
    
    # Save summary CSV
    summary_df = pd.DataFrame(summary_data)
    summary_path = Path('results/metrics/live_2day_model_comparison.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary to {summary_path}")
    
    # Print detailed markdown table
    print("\n\n=== DETAILED MARKDOWN TABLE ===\n")
    for model in MODELS:
        print(f"\n### {model}")
        tickers = sorted(results[model]['per_ticker'].keys())
        print(f"| Ticker | Actual Day1 (₹) | Actual Day2 (₹) | Pred Day1 (₹) | Pred Day2 (₹) | MAPE (%) | DirAcc (%) |")
        print(f"|---|---:|---:|---:|---:|---:|---:|")
        for t in tickers:
            r = results[model]['per_ticker'][t]
            print(f"| {t} | {r['actual'][0]:.2f} | {r['actual'][1]:.2f} | {r['pred'][0]:.2f} | {r['pred'][1]:.2f} | {r['mape']:.2f} | {r['dirac']:.0f} |")

if __name__ == '__main__':
    main()
