import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import DATA_PROC

st.set_page_config(page_title="RetailPulse", page_icon="📈", layout="wide")


@st.cache_data(ttl=300)
def load(name: str):
    path = DATA_PROC / name
    if not path.exists():
        return None
    df = pd.read_parquet(path) if name.endswith(".parquet") else pd.read_csv(path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    return df


st.sidebar.title("RetailPulse")
page = st.sidebar.radio(
    "Page", ["Sales overview", "Forecast explorer", "Customer segments", "Churn risk", "Inventory alerts"]
)

daily = load("daily_sales.parquet")
if daily is None:
    st.error("No processed data found. Run `make all` first.")
    st.stop()

if page == "Sales overview":
    st.header("Sales overview")
    by_day = daily.groupby("Date")[["Quantity", "Revenue"]].sum().reset_index()
    c1, c2, c3 = st.columns(3)
    c1.metric("Total revenue", f"£{daily.Revenue.sum():,.0f}")
    c2.metric("Units sold", f"{daily.Quantity.sum():,.0f}")
    c3.metric("Active SKUs", daily.StockCode.nunique())
    st.plotly_chart(px.line(by_day, x="Date", y="Revenue", title="Daily revenue"), use_container_width=True)
    top = daily.groupby("StockCode").Revenue.sum().nlargest(10).reset_index()
    st.plotly_chart(px.bar(top, x="StockCode", y="Revenue", title="Top 10 SKUs by revenue"), use_container_width=True)

elif page == "Forecast explorer":
    st.header("Forecast explorer")
    fc = load("forecasts.csv")
    if fc is None:
        st.warning("Run `make forecast` first.")
    else:
        sku = st.selectbox("SKU", sorted(fc.StockCode.unique()))
        hist = daily[daily.StockCode == sku].tail(120)[["Date", "Quantity"]].rename(columns={"Quantity": "value"})
        hist["series"] = "actual"
        pred = fc[fc.StockCode == sku][["Date", "forecast"]].rename(columns={"forecast": "value"})
        pred["series"] = "forecast"
        st.plotly_chart(px.line(pd.concat([hist, pred]), x="Date", y="value", color="series"), use_container_width=True)

elif page == "Customer segments":
    st.header("Customer segments (RFM + KMeans)")
    seg = load("segments.csv")
    if seg is None:
        st.warning("Run `make segment` first.")
    else:
        counts = seg.Segment.value_counts().reset_index()
        counts.columns = ["Segment", "Customers"]
        a, b = st.columns(2)
        a.plotly_chart(px.pie(counts, names="Segment", values="Customers"), use_container_width=True)
        b.dataframe(seg.groupby("Segment")[["Recency", "Frequency", "Monetary"]].mean().round(1))
        st.plotly_chart(
            px.scatter(seg, x="Frequency", y="Monetary", color="Segment", log_x=True, log_y=True),
            use_container_width=True,
        )

elif page == "Churn risk":
    st.header("Churn risk")
    sc = load("churn_scores.csv")
    if sc is None:
        st.warning("Run `make churn` first.")
    else:
        thr = st.slider("Minimum churn probability", 0.0, 1.0, 0.7, 0.05)
        risky = sc[sc.churn_probability >= thr]
        st.metric("Customers at risk", len(risky), f"{len(risky) / len(sc):.0%} of base")
        st.dataframe(risky, use_container_width=True)
        st.plotly_chart(px.histogram(sc, x="churn_probability", nbins=30), use_container_width=True)

elif page == "Inventory alerts":
    st.header("Inventory alerts")
    plan = load("inventory_plan.csv")
    if plan is None:
        st.warning("Run `make inventory` first.")
    else:
        status = st.multiselect("Status", plan.status.unique().tolist(), default=["STOCKOUT RISK", "REORDER NOW"])
        st.dataframe(plan[plan.status.isin(status)], use_container_width=True)
