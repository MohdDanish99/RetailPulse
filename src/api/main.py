import pandas as pd
from fastapi import FastAPI, HTTPException

from src.config import DATA_PROC

app = FastAPI(title="RetailPulse API", version="0.1.0")

@app.get("/")
def root():
    return {
        "name": "RetailPulse API",
        "docs": "/docs",
        "endpoints": ["/health", "/forecast/{sku}", "/segment/{customer_id}",
                      "/churn/{customer_id}", "/inventory/{sku}"],
    }

def _read(name: str) -> pd.DataFrame:
    path = DATA_PROC / name
    if not path.exists():
        raise HTTPException(503, f"{name} not found. Run the pipeline first (`make all`).")
    return pd.read_csv(path)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/forecast/{sku}")
def forecast(sku: str, horizon: int = 28):
    df = _read("forecasts.csv")
    df = df[df.StockCode == sku].head(horizon)
    if df.empty:
        raise HTTPException(404, f"No forecast for SKU {sku}")
    return {"sku": sku, "forecast": df[["Date", "forecast"]].to_dict("records")}


@app.get("/segment/{customer_id}")
def segment(customer_id: int):
    df = _read("segments.csv")
    row = df[df.CustomerID == customer_id]
    if row.empty:
        raise HTTPException(404, f"Customer {customer_id} not found")
    return row.iloc[0].to_dict()


@app.get("/churn/{customer_id}")
def churn(customer_id: int):
    df = _read("churn_scores.csv")
    row = df[df.CustomerID == customer_id]
    if row.empty:
        raise HTTPException(404, f"Customer {customer_id} not found")
    p = float(row.iloc[0].churn_probability)
    return {"customer_id": customer_id, "churn_probability": p,
            "risk": "high" if p >= 0.7 else "medium" if p >= 0.4 else "low"}


@app.get("/inventory/{sku}")
def inventory(sku: str):
    df = _read("inventory_plan.csv")
    row = df[df.StockCode == sku]
    if row.empty:
        raise HTTPException(404, f"SKU {sku} not found")
    return row.iloc[0].to_dict()
