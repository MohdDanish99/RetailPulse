import pandas as pd

from src.etl.clean import clean, daily_sales


def _raw():
    return pd.DataFrame(
        {
            "Invoice": ["1", "1", "2", "C3", "4"],
            "StockCode": ["A", "A", "B", "A", "B"],
            "Description": ["x"] * 5,
            "Quantity": [2, 2, 3, -1, 1],
            "InvoiceDate": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-03", "2024-01-04", "2024-01-03"]),
            "Price": [1.0, 1.0, 2.0, 1.0, 0.0],
            "Customer ID": [1.0, 1.0, None, 1.0, 2.0],
            "Country": ["UK"] * 5,
        }
    )


def test_clean_removes_duplicates_and_zero_prices():
    df = clean(_raw())
    assert len(df) == 3  # one duplicate and one zero-price row removed
    assert df["is_return"].sum() == 1


def test_daily_sales_excludes_returns_and_fills_gaps():
    daily = daily_sales(clean(_raw()))
    assert daily["Quantity"].min() >= 0
    assert daily["Date"].nunique() == 3  # Jan 1..3 (return day excluded from range), gaps zero-filled
