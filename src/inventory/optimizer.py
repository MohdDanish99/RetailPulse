import numpy as np
import pandas as pd
from scipy.stats import norm

from src.config import DATA_PROC, INVENTORY_FILE


def safety_stock(daily_std: float, lead_time_days: float, service_level: float = 0.95) -> float:
    return float(norm.ppf(service_level) * daily_std * np.sqrt(lead_time_days))


def reorder_point(mean_daily: float, lead_time_days: float, ss: float) -> float:
    return float(mean_daily * lead_time_days + ss)


def eoq(annual_demand: float, order_cost: float, holding_cost: float) -> float:
    return float(np.sqrt(2 * annual_demand * order_cost / holding_cost))


def build_plan(daily, forecasts, inventory, service_level=0.95, order_cost=50.0, holding_rate=0.2):
    inv = inventory.set_index("StockCode")
    rows = []
    for sku, f in forecasts.groupby("StockCode"):
        if sku not in inv.index:
            continue
        hist = daily[daily.StockCode == sku].sort_values("Date").tail(90)["Quantity"]
        mean_d, sigma = float(f["forecast"].mean()), float(hist.std())
        lt, stock, cost = inv.loc[sku, "lead_time_days"], inv.loc[sku, "stock_on_hand"], inv.loc[sku, "unit_cost"]
        ss = safety_stock(sigma, lt, service_level)
        rop = reorder_point(mean_d, lt, ss)
        if stock < mean_d * lt:
            status = "STOCKOUT RISK"
        elif stock <= rop:
            status = "REORDER NOW"
        else:
            status = "OK"
        rows.append(
            {
                "StockCode": sku,
                "forecast_daily_demand": round(mean_d, 2),
                "stock_on_hand": int(stock),
                "days_of_cover": round(stock / mean_d, 1) if mean_d > 0 else None,
                "lead_time_days": int(lt),
                "safety_stock": round(ss),
                "reorder_point": round(rop),
                "eoq": round(eoq(mean_d * 365, order_cost, holding_rate * cost)) if mean_d > 0 else 0,
                "status": status,
            }
        )
    order = {"STOCKOUT RISK": 0, "REORDER NOW": 1, "OK": 2}
    return pd.DataFrame(rows).sort_values("status", key=lambda s: s.map(order)).reset_index(drop=True)


def main():
    daily = pd.read_parquet(DATA_PROC / "daily_sales.parquet")
    forecasts = pd.read_csv(DATA_PROC / "forecasts.csv")
    inventory = pd.read_csv(INVENTORY_FILE)
    plan = build_plan(daily, forecasts, inventory)
    plan.to_csv(DATA_PROC / "inventory_plan.csv", index=False)
    print(plan["status"].value_counts().to_string())
    print(f"Wrote {len(plan)} rows -> data/processed/inventory_plan.csv")


if __name__ == "__main__":
    main()
