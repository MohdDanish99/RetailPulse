from pathlib import Path

import pandas as pd

from src.config import RAW_FILE


def load_raw(path=RAW_FILE) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} not found. Run `make data` for synthetic data, or place the "
            "Online Retail II CSV there."
        )
    return pd.read_csv(path, parse_dates=["InvoiceDate"])


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Dedupe, drop invalid prices, flag returns, add revenue and date columns."""
    df = df.drop_duplicates().rename(columns={"Customer ID": "CustomerID"}).copy()
    df = df.dropna(subset=["InvoiceDate"])
    df = df[df["Price"] > 0].copy()
    df["Invoice"] = df["Invoice"].astype(str)
    df["StockCode"] = df["StockCode"].astype(str)
    df["CustomerID"] = df["CustomerID"].astype("Int64")
    df["is_return"] = df["Invoice"].str.startswith("C")
    df["Revenue"] = df["Quantity"] * df["Price"]
    df["Date"] = df["InvoiceDate"].dt.normalize()
    return df.reset_index(drop=True)


def daily_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Daily units and revenue per SKU, with zero-filled gaps (needed for forecasting)."""
    sales = df[(~df["is_return"]) & (df["Quantity"] > 0)]
    g = sales.groupby(["Date", "StockCode"], as_index=False).agg(
        Quantity=("Quantity", "sum"), Revenue=("Revenue", "sum")
    )
    full = pd.MultiIndex.from_product(
        [pd.date_range(g["Date"].min(), g["Date"].max()), g["StockCode"].unique()],
        names=["Date", "StockCode"],
    )
    return g.set_index(["Date", "StockCode"]).reindex(full, fill_value=0).reset_index()
