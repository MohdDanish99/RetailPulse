"""Churn = no purchase in the 90 days after a cutoff. Features use only data BEFORE the cutoff."""
import joblib
import mlflow
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import CHURN_DAYS, DATA_PROC, MODELS

FEATURES = [
    "recency", "frequency", "monetary", "tenure", "avg_order_value",
    "orders_last_30d", "orders_last_90d", "avg_gap_days",
]


def build_features(df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    h = df[(df["InvoiceDate"] < cutoff) & df["CustomerID"].notna() & ~df["is_return"]]
    f = h.groupby("CustomerID").agg(
        frequency=("Invoice", "nunique"),
        monetary=("Revenue", "sum"),
        first=("InvoiceDate", "min"),
        last=("InvoiceDate", "max"),
    )
    f["recency"] = (cutoff - f["last"]).dt.days
    f["tenure"] = (cutoff - f["first"]).dt.days
    f["avg_order_value"] = f["monetary"] / f["frequency"]
    f["avg_gap_days"] = f["tenure"] / f["frequency"]
    for w in (30, 90):
        recent = h[h["InvoiceDate"] >= cutoff - pd.Timedelta(days=w)].groupby("CustomerID")["Invoice"].nunique()
        f[f"orders_last_{w}d"] = recent.reindex(f.index).fillna(0)
    return f[FEATURES]


def build_labels(df, cutoff, end, index) -> pd.Series:
    fut = df[(df["InvoiceDate"] >= cutoff) & (df["InvoiceDate"] < end) & df["CustomerID"].notna() & ~df["is_return"]]
    active = fut["CustomerID"].unique()
    return pd.Series((~index.isin(active)).astype(int), index=index, name="churned")


def metrics(y, proba) -> dict:
    pred = proba >= 0.5
    return {
        "roc_auc": roc_auc_score(y, proba),
        "pr_auc": average_precision_score(y, proba),
        "recall": recall_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
    }


def main():
    df = pd.read_parquet(DATA_PROC / "transactions.parquet")
    last_date = df["InvoiceDate"].max().normalize()
    cutoff = last_date - pd.Timedelta(days=CHURN_DAYS)
    X = build_features(df, cutoff)
    y = build_labels(df, cutoff, last_date + pd.Timedelta(days=1), X.index)
    print(f"{len(X)} customers | churn rate {y.mean():.1%}")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

    models = {
        "logreg_baseline": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=300, random_state=42),
    }
    mlflow.set_experiment("retailpulse-churn")
    MODELS.mkdir(exist_ok=True)
    best_name, best_pr, best_model = None, -1, None
    for name, model in models.items():
        model.fit(Xtr, ytr)
        m = metrics(yte, model.predict_proba(Xte)[:, 1])
        with mlflow.start_run(run_name=name):
            mlflow.log_params({"model": name, "churn_days": CHURN_DAYS, "cutoff": str(cutoff.date())})
            mlflow.log_metrics({**m, "positive_rate": float(yte.mean())})
        print(name, {k: round(v, 3) for k, v in m.items()}, f"(PR-AUC chance level = {yte.mean():.3f})")
        if m["pr_auc"] > best_pr:
            best_name, best_pr, best_model = name, m["pr_auc"], model

    path = MODELS / "churn_model.joblib"
    joblib.dump(best_model, path)
    with mlflow.start_run(run_name=f"best_{best_name}"):
        mlflow.log_artifact(str(path))
    print(f"Best: {best_name} (PR-AUC {best_pr:.3f}) -> {path}")

    # Score every customer as of today
    X_now = build_features(df, last_date + pd.Timedelta(days=1))
    scores = pd.DataFrame({"CustomerID": X_now.index, "churn_probability": best_model.predict_proba(X_now)[:, 1].round(4)})
    scores.sort_values("churn_probability", ascending=False).to_csv(DATA_PROC / "churn_scores.csv", index=False)
    print(f"Scored {len(scores)} customers -> data/processed/churn_scores.csv")


if __name__ == "__main__":
    main()
