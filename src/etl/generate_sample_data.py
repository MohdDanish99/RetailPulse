"""Generate a synthetic dataset shaped like Kaggle's 'Online Retail II'.

Lets you run the whole pipeline without downloading anything.
Replace data/raw/online_retail_II.csv with the real file when ready.
"""
import numpy as np
import pandas as pd

from src.config import DATA_RAW, INVENTORY_FILE, RAW_FILE


def generate(n_customers=800, n_products=60, start="2010-01-01", end="2011-12-09", seed=42):
    rng = np.random.default_rng(seed)
    days = pd.date_range(start, end)
    products = pd.DataFrame(
        {
            "StockCode": [f"P{i:03d}" for i in range(n_products)],
            "Description": [f"PRODUCT {i:03d}" for i in range(n_products)],
            "Price": rng.uniform(1, 30, n_products).round(2),
        }
    )
    popularity = rng.dirichlet(np.ones(n_products) * 0.6)
    # Weekly + Q4 seasonality
    day_factor = (1.0 + 0.6 * days.month.isin([10, 11, 12])) * np.where(days.dayofweek == 5, 0.5, 1.0)

    rows = []
    invoice_no = 500000
    for i in range(n_customers):
        cust_id = 12000 + i
        rate = rng.uniform(0.01, 0.10)
        first = int(rng.integers(0, 250))
        churns = rng.random() < 0.35
        last = min(len(days) - 1, first + int(rng.integers(60, 400))) if churns else len(days) - 1
        idx = np.arange(first, last + 1)
        buys = idx[rng.random(len(idx)) < np.clip(rate * day_factor[idx], 0, 1)]
        for d in buys:
            invoice_no += 1
            ts = days[d] + pd.Timedelta(hours=int(rng.integers(8, 19)), minutes=int(rng.integers(0, 60)))
            cid = np.nan if rng.random() < 0.05 else cust_id  # guest checkouts
            for s in rng.choice(n_products, size=int(rng.integers(1, 6)), replace=False, p=popularity):
                qty = int(rng.integers(1, 12))
                p = products.iloc[s]
                rows.append((str(invoice_no), p.StockCode, p.Description, qty, ts, p.Price, cid, "United Kingdom"))
                if rng.random() < 0.02:  # returns use a 'C' invoice prefix and negative quantity
                    rows.append((f"C{invoice_no}", p.StockCode, p.Description, -qty,
                                 ts + pd.Timedelta(days=2), p.Price, cid, "United Kingdom"))

    cols = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"]
    df = pd.DataFrame(rows, columns=cols)
    df = pd.concat([df, df.sample(50, random_state=seed)])  # inject duplicates for the ETL to remove

    sold = df[df.Quantity > 0].groupby("StockCode").Quantity.sum()
    avg_daily = sold.reindex(products.StockCode).fillna(1).values / len(days)
    inventory = pd.DataFrame(
        {
            "StockCode": products.StockCode,
            "stock_on_hand": (avg_daily * rng.uniform(1, 25, n_products)).round().astype(int),
            "lead_time_days": rng.integers(3, 15, n_products),
            "unit_cost": (products.Price * 0.6).round(2),
        }
    )
    return df, inventory


def main():
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    df, inventory = generate()
    df.to_csv(RAW_FILE, index=False)
    inventory.to_csv(INVENTORY_FILE, index=False)
    print(f"Wrote {len(df):,} transaction rows -> {RAW_FILE}")
    print(f"Wrote {len(inventory)} inventory rows -> {INVENTORY_FILE}")


if __name__ == "__main__":
    main()
