import glob
import pandas as pd
import numpy as np
from pathlib import Path

FORECAST_DIR = Path('results/forecasts')
METRICS_CSV = Path('results/metrics/model_comparison.csv')

def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))

def mape(a, b):
    # avoid divide by zero
    mask = a != 0
    if not mask.any():
        return np.nan
    return np.mean(np.abs((a[mask] - b[mask]) / a[mask])) * 100

def directional_accuracy(actual, pred):
    # actual: series of actual prices, pred: series of predicted prices aligned to same dates
    # compute direction of actual move: actual.diff()
    actual_diff = actual.diff().fillna(0)
    pred_diff = pred - actual.shift(1).fillna(actual.iloc[0])
    # sign of movement
    act_sign = np.sign(actual_diff)
    pred_sign = np.sign(pred_diff)
    # directional accuracy: fraction of times signs match (exclude zeros where no change?)
    mask = (act_sign != 0)
    if not mask.any():
        return np.nan
    return (act_sign[mask] == pred_sign[mask]).mean() * 100


def compute_for_file(fp):
    df = pd.read_csv(fp, parse_dates=['Date'])
    # find predicted column (anything with 'arima' or 'ARIMA' or endswith '_Pred')
    pred_col = None
    for c in df.columns:
        if 'arima' in c.lower() or 'pred' in c.lower():
            pred_col = c
            break
    if pred_col is None:
        raise ValueError(f'No prediction column in {fp}')
    actual = df['Actual'].astype(float)
    pred = df[pred_col].astype(float)

    return rmse(actual.values, pred.values), mape(actual.values, pred.values), directional_accuracy(actual, pred)


def main():
    arima_files = sorted(glob.glob(str(FORECAST_DIR / '*_arima.csv')))
    metrics = {}
    for f in arima_files:
        ticker = Path(f).stem.replace('_arima','').replace('_','.')
        try:
            r, m, d = compute_for_file(f)
            metrics[ticker] = {'RMSE': round(float(r),4), 'MAPE': round(float(m),4), 'DirAcc': round(float(d),4)}
            print(f'{ticker}: RMSE={r:.4f}, MAPE={m:.4f}, DirAcc={d:.2f}%')
        except Exception as e:
            print(f'Failed {f}: {e}')

    if not METRICS_CSV.exists():
        print('Metrics CSV not found at', METRICS_CSV)
        return

    dfm = pd.read_csv(METRICS_CSV)
    # Update ARIMA rows
    for idx, row in dfm.iterrows():
        if str(row.get('Model','')).strip().upper() == 'ARIMA':
            t = row['Ticker']
            if t in metrics:
                dfm.at[idx, 'RMSE'] = metrics[t]['RMSE']
                dfm.at[idx, 'MAPE'] = metrics[t]['MAPE']
                dfm.at[idx, 'DirAcc'] = metrics[t]['DirAcc']
    dfm.to_csv(METRICS_CSV, index=False)
    print('Updated', METRICS_CSV)

if __name__ == '__main__':
    main()
