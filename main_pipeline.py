from src.data.fetch_data import fetch_all, STOCK_UNIVERSE, SECTOR_MAP
from src.data.screening import screening_report

data = fetch_all()

report = screening_report(
    data=data,
    sector_map=SECTOR_MAP,
    stock_names=STOCK_UNIVERSE
)

print(report.head())