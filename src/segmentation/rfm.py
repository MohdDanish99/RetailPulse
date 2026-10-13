import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.config import DATA_PROC

NAMES = ["Hibernating", "At-Risk", "Loyal", "Champions"]  # ordered low -> high value


def build_rfm(df: pd.DataFrame) -> pd.DataFrame:
    sales = df[df["CustomerID"].notna() & ~df["is_return"]]
    snapshot = sales["InvoiceDate"].max() + pd.Timedelta(days=1)
    return sales.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda s: (snapshot - s.max()).days),
        Frequency=("Invoice", "nunique"),
        Monetary=("Revenue", "sum"),
    )


def segment(rfm: pd.DataFrame, k: int = 4, seed: int = 42) -> pd.DataFrame:
    X = StandardScaler().fit_transform(np.log1p(rfm[["Recency", "Frequency", "Monetary"]].clip(lower=0)))
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
    c = km.cluster_centers_
    value = c[:, 1] + c[:, 2] - c[:, 0]  # high frequency + spend, low recency = better
    rank = {cl: r for r, cl in enumerate(np.argsort(value))}
    names = NAMES if k == len(NAMES) else [f"Tier {i + 1}" for i in range(k)]
    out = rfm.copy()
    out["Segment"] = [names[rank[cl]] for cl in km.labels_]
    out.attrs["silhouette"] = silhouette_score(X, km.labels_, sample_size=min(2000, len(X)), random_state=seed)
    return out


def silhouette_by_k(rfm: pd.DataFrame, ks=range(2, 8), seed: int = 42) -> dict:
    X = StandardScaler().fit_transform(np.log1p(rfm[["Recency", "Frequency", "Monetary"]].clip(lower=0)))
    return {
        k: silhouette_score(X, KMeans(k, n_init=10, random_state=seed).fit_predict(X)) for k in ks
    }


def main():
    df = pd.read_parquet(DATA_PROC / "transactions.parquet")
    rfm = build_rfm(df)
    print("silhouette by k:", {k: round(v, 3) for k, v in silhouette_by_k(rfm).items()})
    seg = segment(rfm)
    seg.reset_index().to_csv(DATA_PROC / "segments.csv", index=False)
    print(f"silhouette (k=4): {seg.attrs['silhouette']:.3f}")
    print(seg.groupby("Segment")[["Recency", "Frequency", "Monetary"]].mean().round(1))


if __name__ == "__main__":
    main()
