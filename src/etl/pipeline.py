from src.config import DATA_PROC
from src.etl.clean import clean, daily_sales, load_raw


def run():
    DATA_PROC.mkdir(parents=True, exist_ok=True)
    raw = load_raw()
    df = clean(raw)
    daily = daily_sales(df)
    df.to_parquet(DATA_PROC / "transactions.parquet", index=False)
    daily.to_parquet(DATA_PROC / "daily_sales.parquet", index=False)
    print(f"raw rows: {len(raw):,} -> clean rows: {len(df):,}")
    print(f"daily SKU rows: {len(daily):,} | date range: {daily.Date.min().date()} to {daily.Date.max().date()}")


if __name__ == "__main__":
    run()
