"""
main_pipeline.py
----------------
End-to-end orchestrator for all capstone tasks.

Usage:
    python main_pipeline.py                   # run everything
    python main_pipeline.py --task 1 2 3      # run specific tasks
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("pipeline")


# ── Task 1: Stock Selection & Screening ──────────────────────────────────────

def task1_screening():
    log.info("=" * 60)
    log.info("TASK 1: Stock Universe Selection & Screening")
    log.info("=" * 60)

    from src.data.fetch_data import fetch_all, STOCK_UNIVERSE, SECTOR_MAP
    from src.data.screening import screening_report

    data = fetch_all()
    report = screening_report(
        data=data,
        sector_map=SECTOR_MAP,
        stock_names=STOCK_UNIVERSE,
    )
    print("\n" + report.to_string() + "\n")
    return data


# ── Task 2: Data Preprocessing ──────────────────────────────────────────────

def task2_preprocess(data=None):
    log.info("=" * 60)
    log.info("TASK 2: Data Preprocessing")
    log.info("=" * 60)

    from src.data.fetch_data import load_all_raw
    from src.data.preprocess import preprocess_all, stationarity_report

    if data is None:
        data = load_all_raw()

    processed = preprocess_all(data)

    # Print stationarity report
    stat_report = stationarity_report(data)
    print("\nStationarity Report:")
    print(stat_report.to_string() + "\n")

    for ticker, d in processed.items():
        log.info(f"  {ticker}: train={len(d['train'])}  test={len(d['test'])}  d_order={d['d_order']}")

    return processed


# ── Task 3: Time Series Forecasting ─────────────────────────────────────────

def task3_forecasting(processed):
    log.info("=" * 60)
    log.info("TASK 3: Time Series Forecasting")
    log.info("=" * 60)

    all_metrics = []

    # ARIMA
    log.info("─── ARIMA ───")
    from src.models.arima import run_arima_pipeline
    arima_preds, arima_fc, arima_met = run_arima_pipeline(processed)
    all_metrics.extend(arima_met)

    # Prophet
    log.info("─── Prophet ───")
    from src.models.prophet_model import run_prophet_pipeline
    prophet_preds, prophet_fc, prophet_met = run_prophet_pipeline(processed)
    all_metrics.extend(prophet_met)

    # LSTM
    log.info("─── LSTM ───")
    from src.models.lstm import run_lstm_pipeline
    lstm_preds, lstm_fc, lstm_met = run_lstm_pipeline(processed, epochs=30)
    all_metrics.extend(lstm_met)

    # GRU
    log.info("─── GRU ───")
    from src.models.lstm import run_gru_pipeline
    gru_preds, gru_fc, gru_met = run_gru_pipeline(processed, epochs=30)
    all_metrics.extend(gru_met)

    # Transformer
    log.info("─── Transformer ───")
    from src.models.transformer import run_transformer_pipeline
    trans_preds, trans_fc, trans_met = run_transformer_pipeline(processed, epochs=30)
    all_metrics.extend(trans_met)

    # Save combined metrics
    metrics_df = pd.DataFrame(all_metrics)
    out_dir = ROOT / "results" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(out_dir / "model_comparison.csv", index=False)
    log.info(f"Model comparison saved → {out_dir / 'model_comparison.csv'}")
    print("\n" + metrics_df.to_string() + "\n")

    return {
        "arima":       {"preds": arima_preds,   "fc": arima_fc},
        "prophet":     {"preds": prophet_preds, "fc": prophet_fc},
        "lstm":        {"preds": lstm_preds,    "fc": lstm_fc},
        "gru":         {"preds": gru_preds,     "fc": gru_fc},
        "transformer": {"preds": trans_preds,   "fc": trans_fc},
    }, metrics_df


# ── Task 4: Volatility & Trend Analysis ─────────────────────────────────────

def task4_analysis(data=None):
    log.info("=" * 60)
    log.info("TASK 4: Volatility & Trend Analysis")
    log.info("=" * 60)

    from src.data.fetch_data import load_all_raw
    from src.analysis.volatility import volatility_summary
    from src.analysis.trend_analysis import trend_summary
    from src.analysis.correlation import return_correlation_matrix, correlation_weights

    if data is None:
        data = load_all_raw()

    vol_df   = volatility_summary(data)
    trend_df = trend_summary(data)
    corr_df  = return_correlation_matrix(data)
    corr_w   = correlation_weights(corr_df)

    print("\nVolatility Summary:")
    print(vol_df.to_string())
    print("\nTrend Summary:")
    print(trend_df.to_string())
    print("\nCorrelation Matrix:")
    print(corr_df.round(2).to_string())
    print()

    return vol_df, trend_df, corr_df, corr_w


# ── Task 5: Portfolio Construction ───────────────────────────────────────────

def task5_portfolio(model_results, vol_df, corr_w, data=None):
    log.info("=" * 60)
    log.info("TASK 5: Portfolio Construction & Allocation")
    log.info("=" * 60)

    from src.data.fetch_data import STOCK_UNIVERSE, SECTOR_MAP, load_all_raw
    from src.data.screening import sector_momentum
    from src.portfolio.allocation import build_portfolio

    if data is None:
        data = load_all_raw()

    # Use the best model's forecasts (or ARIMA as default)
    best_model = "arima"
    future_fc  = model_results[best_model]["fc"]

    # Current prices = last available close
    current_prices = {}
    for ticker, df in data.items():
        close = df["Close"].squeeze().dropna()
        current_prices[ticker] = float(close.iloc[-1])

    # Volatility estimates from GARCH
    vol_estimates = vol_df["GARCH_FcstVol"].to_dict()

    # Sector momentum
    sec_mom = sector_momentum(data, SECTOR_MAP)

    tickers = list(STOCK_UNIVERSE.keys())
    alloc = build_portfolio(
        tickers=tickers,
        sector_map=SECTOR_MAP,
        stock_names=STOCK_UNIVERSE,
        future_forecasts=future_fc,
        current_prices=current_prices,
        volatility_estimates=vol_estimates,
        corr_weights=corr_w,
        sector_returns=sec_mom,
    )

    print("\nPortfolio Allocation:")
    print(alloc.to_string())
    print()

    return alloc


# ── Task 6: Model Comparison ────────────────────────────────────────────────

def task6_comparison(metrics_df):
    log.info("=" * 60)
    log.info("TASK 6: Model Comparison")
    log.info("=" * 60)

    print("\n── Cross-Model Evaluation ──")
    print(metrics_df.to_string())

    # Average across tickers per model
    avg = metrics_df.groupby("Model")[["RMSE", "MAPE", "DirAcc"]].mean().round(4)
    print("\n── Average Metrics by Model ──")
    print(avg.to_string())

    best = avg["MAPE"].idxmin()
    log.info(f"\nBest model by average MAPE: {best}")
    return best


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TSA 2026 Pipeline")
    parser.add_argument("--task", nargs="*", type=int, default=None,
                        help="Tasks to run (1-6). Default: all")
    args = parser.parse_args()

    tasks = args.task or [1, 2, 3, 4, 5, 6]
    data = None
    processed = None
    model_results = None
    metrics_df = None
    vol_df = None
    corr_w = None

    if 1 in tasks:
        data = task1_screening()

    if 2 in tasks:
        processed = task2_preprocess(data)

    if 3 in tasks:
        if processed is None:
            processed = task2_preprocess(data)
        model_results, metrics_df = task3_forecasting(processed)
        # Save metrics deliverable
        metrics_df.to_csv("deliverables/task3_forecasting/model_metrics_comparison.csv", index=False)
    else:
        # Try to load existing results from disk for Task 5/6
        try:
            import pandas as pd
            metrics_path = Path("results/metrics/model_comparison.csv")
            if metrics_path.exists():
                metrics_df = pd.read_csv(metrics_path)
                log.info("Loaded existing metrics from disk.")
                
                # Load forecasts
                arima_fc = {}
                fc_dir = Path("results/forecasts")
                for f in fc_dir.glob("*_arima.csv"):
                    # Filename example: TCS_NS_arima.csv -> TCS.NS
                    ticker = f.name.replace("_arima.csv", "").replace("_NS", ".NS")
                    df_fc = pd.read_csv(f)
                    # The column in these individual files is named "ARIMA_Pred" or "Forecast"
                    # But for build_portfolio, we just need the values.
                    if "ARIMA_Pred" in df_fc.columns:
                        arima_fc[ticker] = df_fc["ARIMA_Pred"].values
                    elif "Forecast" in df_fc.columns:
                        arima_fc[ticker] = df_fc["Forecast"].values
                    else:
                        # Fallback to the last column
                        arima_fc[ticker] = df_fc.iloc[:, -1].values
                
                model_results = {'arima': {'fc': arima_fc}}
                log.info(f"Loaded forecasts for {len(arima_fc)} stocks: {list(arima_fc.keys())}")
        except Exception as e:
            log.warning(f"Could not load existing results: {e}")

    # Task 4: Analysis
    if 4 in tasks:
        vol_df, trend_df, corr_df, corr_w = task4_analysis(data)
        # Note: Analysis modules now save their own deliverables internally

    # Task 5: Portfolio
    if 5 in tasks:
        if model_results is None:
            log.warning("Task 5 requires Task 3 results. Running Task 3 first.")
            if processed is None:
                processed = task2_preprocess(data)
            model_results, metrics_df = task3_forecasting(processed)
        if vol_df is None:
            vol_df, trend_df, corr_df, corr_w = task4_analysis(data)
        
        # Build portfolio (now saves to deliverables/task5_portfolio/)
        from src.data.fetch_data import STOCK_UNIVERSE, SECTOR_MAP
        from src.data.screening import sector_momentum
        
        # Get current prices from data
        current_prices = {t: df['Close'].iloc[-1] for t, df in data.items()}
        vol_estimates = vol_df['GARCH_FcstVol'].to_dict()
        sec_mom = sector_momentum(data, SECTOR_MAP)
        
        from src.portfolio.allocation import build_portfolio
        alloc = build_portfolio(
            tickers=list(data.keys()),
            sector_map=SECTOR_MAP,
            stock_names=STOCK_UNIVERSE,
            future_forecasts=model_results['arima']['fc'], # Use the dictionary of arrays
            current_prices=current_prices,
            volatility_estimates=vol_estimates,
            corr_weights=corr_w,
            sector_returns=sec_mom
        )

    # Task 6: Comparison
    if 6 in tasks:
        if metrics_df is None:
            log.warning("Task 6 requires Task 3 metrics. Run Task 3 first.")
        else:
            metrics_df.to_csv("deliverables/task6_comparison/final_model_performance.csv", index=False)
            
            # Generate Report Figures
            try:
                from src.utils.visualization import plot_screening_results, plot_portfolio_summary, plot_forecast_comparison
                from src.data.screening import screening_report
                
                log.info("Generating report figures...")
                # 1. Screening plots
                report = screening_report(data)
                plot_screening_results(report)
                
                # 2. Portfolio plots
                plot_portfolio_summary(alloc, corr_df)
                
                # 3. Forecast plots for top 3 stocks
                for ticker in list(data.keys())[:3]:
                    actual = processed[ticker]['test']
                    preds = {m: model_results[m.lower()]['preds'][ticker] for m in ['ARIMA', 'Prophet', 'LSTM', 'GRU', 'Transformer']}
                    plot_forecast_comparison(ticker, actual, preds)
                    
            except Exception as e:
                log.error(f"Visualization failed: {e}")

    log.info("Pipeline complete. Consolidating deliverables...")
    consolidate_deliverables()
    log.info("Check the 'deliverables/' folder for all task outputs.")

def consolidate_deliverables():
    import shutil
    from pathlib import Path
    
    base = Path("deliverables")
    mapping = {
        "reports/tables/stock_screening_report.csv": base / "task1_screening/screening_report.csv",
        "data/processed/": base / "task2_preprocessing/",
        "results/forecasts/": base / "task3_forecasting/",
        "reports/tables/volatility_summary.csv": base / "task4_analysis/volatility_metrics.csv",
        "reports/tables/trend_metrics.csv": base / "task4_analysis/trend_metrics.csv",
        "deliverables/task4_analysis/correlation_matrix.csv": base / "task4_analysis/correlation_matrix.csv",
        "results/portfolio/portfolio_allocation.csv": base / "task5_portfolio/portfolio_allocation.csv",
        "results/metrics/model_comparison.csv": base / "task6_comparison/final_comparison.csv",
    }
    
    for src_path, dest_path in mapping.items():
        src = Path(src_path)
        if src.exists():
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                if dest_path.exists(): shutil.rmtree(dest_path)
                shutil.copytree(src, dest_path)
            else:
                shutil.copy2(src, dest_path)

if __name__ == "__main__":
    main()