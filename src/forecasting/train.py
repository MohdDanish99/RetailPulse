"""Compare forecasters on a time-based holdout, log to MLflow, write future forecasts."""
import argparse

import mlflow
import numpy as np
import pandas as pd

from src.config import DATA_PROC, FORECAST_HORIZON, TOP_N_SKUS
from src.forecasting.baseline import rmse, seasonal_naive, wape


def get_series(daily: pd.DataFrame, sku=None) -> pd.Series:
    d = daily if sku is None else daily[daily.StockCode == sku]
    return d.groupby("Date")["Quantity"].sum().asfreq("D", fill_value=0)


def available_models(use_lstm: bool) -> dict:
    models = {"seasonal_naive": lambda s, h: seasonal_naive(s.values, h)}
    try:
        from src.forecasting.prophet_model import prophet_forecast

        import prophet  # noqa: F401

        models["prophet"] = prophet_forecast
    except ImportError:
        print("prophet not installed; skipping")
    if use_lstm:
        from src.forecasting.lstm_model import lstm_forecast

        models["lstm"] = lstm_forecast
    return models


def main(use_lstm: bool = False):
    h = FORECAST_HORIZON
    daily = pd.read_parquet(DATA_PROC / "daily_sales.parquet")
    models = available_models(use_lstm)
    top = daily.groupby("StockCode").Quantity.sum().nlargest(TOP_N_SKUS).index.tolist()

    mlflow.set_experiment("retailpulse-forecasting")
    total = get_series(daily)
    results = {}
    for name, fn in models.items():
        train, test = total.iloc[:-h], total.iloc[-h:]
        pred = fn(train, h)
        results[name] = wape(test.values, pred)
        with mlflow.start_run(run_name=f"total_{name}"):
            mlflow.log_params({"model": name, "target": "total_units", "horizon": h})
            mlflow.log_metrics({"wape": results[name], "rmse": rmse(test.values, pred)})
        print(f"[total demand] {name:15s} WAPE={results[name]:.3f}")

    best = min(results, key=results.get)
    base = results["seasonal_naive"]
    print(f"Best model: {best} ({(1 - results[best] / base) * 100:.1f}% better WAPE than baseline)")

    rows = []
    for sku in top:
        s = get_series(daily, sku)
        pred = models[best](s, h)
        dates = pd.date_range(s.index[-1] + pd.Timedelta(days=1), periods=h)
        rows.append(pd.DataFrame({"Date": dates, "StockCode": sku, "forecast": np.round(pred, 2)}))
    pd.concat(rows).to_csv(DATA_PROC / "forecasts.csv", index=False)
    print(f"Wrote {h}-day forecasts for {len(top)} SKUs -> data/processed/forecasts.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lstm", action="store_true", help="also train the LSTM (needs tensorflow)")
    main(ap.parse_args().lstm)
