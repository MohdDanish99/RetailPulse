"""Lightweight drift detection using the Population Stability Index (PSI).
Rule of thumb: <0.1 stable, 0.1-0.2 watch, >0.2 significant drift -> retrain."""
import numpy as np
import pandas as pd

from src.churn.train import FEATURES, build_features
from src.config import DATA_PROC


def psi(expected, actual, bins: int = 10) -> float:
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.clip(np.histogram(expected, edges)[0] / len(expected), 1e-6, None)
    a = np.clip(np.histogram(actual, edges)[0] / len(actual), 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def detect_drift(ref: pd.DataFrame, cur: pd.DataFrame, columns, threshold: float = 0.2) -> pd.DataFrame:
    rows = [{"feature": c, "psi": round(psi(ref[c], cur[c]), 4)} for c in columns]
    out = pd.DataFrame(rows)
    out["drifted"] = out["psi"] > threshold
    return out


def main():
    df = pd.read_parquet(DATA_PROC / "transactions.parquet")
    last = df["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)
    ref = build_features(df, last - pd.Timedelta(days=180))
    cur = build_features(df, last)
    report = detect_drift(ref, cur, FEATURES)
    report.to_csv(DATA_PROC / "drift_report.csv", index=False)
    print(report.to_string(index=False))
    if report["drifted"].any():
        print("\nDrift detected -> trigger retraining.")


if __name__ == "__main__":
    main()
