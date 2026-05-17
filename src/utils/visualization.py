import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

IMG_DIR = Path(__file__).resolve().parents[2] / "deliverables" / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

def plot_screening_results(screening_df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    screening_df['Mean_AnnVol'].sort_values().plot(kind='barh', ax=axes[0], color='skyblue')
    axes[0].set_title('Annualized Volatility by Stock')

    sns.scatterplot(data=screening_df, x='Trend_Strength', y='6M_CumReturn', hue='Sector', s=100, ax=axes[1])
    axes[1].set_title('Trend Strength vs 6-Month Momentum')
    
    plt.tight_layout()
    plt.savefig(IMG_DIR / "task1_screening_analysis.png", dpi=300)
    plt.close()

def plot_forecast_comparison(ticker: str, actual: pd.Series, predictions: dict):
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label='Actual Price', color='black', linewidth=2)
    
    colors = {'ARIMA': 'red', 'Prophet': 'orange', 'LSTM': 'blue', 'GRU': 'green', 'Transformer': 'purple'}
    
    for model_name, pred_series in predictions.items():
        plt.plot(pred_series, label=f'{model_name} Forecast', linestyle='--', alpha=0.8, color=colors.get(model_name))
        
    plt.title(f'Model Comparison: {ticker} (Test Period)')
    plt.legend()
    plt.savefig(IMG_DIR / f"task3_forecast_{ticker.replace('.', '_')}.png", dpi=300)
    plt.close()

def plot_portfolio_summary(alloc_df: pd.DataFrame, corr_matrix: pd.DataFrame):
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn', center=0)
    plt.title('Stock Return Correlations')
    plt.savefig(IMG_DIR / "task4_correlation_matrix.png", dpi=300)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    alloc_df.set_index('Name')['Weight'].plot(kind='pie', autopct='%1.1f%%', ax=axes[0])
    axes[0].set_ylabel('')
    axes[0].set_title('Portfolio Allocation by Weight')

    if 'Amount' in alloc_df.columns:
        alloc_df.groupby('Sector')['Amount'].sum().plot(kind='bar', ax=axes[1], color='coral')
        axes[1].set_title('Capital Exposure by Sector (₹)')
    
    plt.tight_layout()
    plt.savefig(IMG_DIR / "task5_portfolio_allocation.png", dpi=300)
    plt.close()
