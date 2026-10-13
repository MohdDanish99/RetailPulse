# RetailPulse: AI-Powered Customer Analytics & Demand Forecasting

Analyzes sales, customer and inventory data to deliver **demand forecasting, customer segmentation,
churn prediction and inventory optimization**, with an API, dashboard and MLOps tooling.

## Architecture

```
Raw CSV -> ETL (clean, dedupe, returns, daily SKU aggregation) -> data/processed/*.parquet
                |
   +------------+-------------+--------------------+
Forecasting   Segmentation   Churn model      Inventory optimizer
(baseline,    (RFM+KMeans)   (LogReg vs       (safety stock,
 Prophet,                     GradBoosting)    reorder point, EOQ)
 LSTM)
   +------------+-------------+--------------------+
                |
          MLflow tracking  ->  FastAPI  +  Streamlit  ->  Docker -> Kubernetes
                |
        PSI drift detection (retrain trigger)
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
make setup
make all          # synthetic data -> ETL -> forecast -> segments -> churn -> inventory -> drift
make dashboard    # http://localhost:8501
make api          # http://localhost:8000/docs
make mlflow       # http://localhost:5000
make test
```

### Using the real dataset
Download **Online Retail II** from Kaggle, export it as CSV to `data/raw/online_retail_II.csv`
(columns: Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country),
then skip `make data` and run `make etl forecast segment churn`.
The real data has no inventory table, so keep the generated `data/raw/inventory.csv` as a placeholder
or replace it with your own (StockCode, stock_on_hand, lead_time_days, unit_cost).

## Modeling notes
- **Forecasting:** time-based holdout (last 28 days), never random. Metric is WAPE (safe with zero-demand days) and RMSE. Every model is compared against a seasonal-naive baseline.
- **Churn:** churn = no purchase in the next 90 days. Features come only from data before the cutoff (no leakage). Judge with PR-AUC and recall, compared with the churn base rate.
- **Segmentation:** RFM, log-scaled, KMeans (k=4), with silhouette scores for choosing k.
- **Inventory:** safety stock = z * sigma_daily * sqrt(lead time); ROP = demand * lead time + safety stock; EOQ.
- **Drift:** PSI per feature (>0.2 triggers retraining).

## Results (fill in after running on your data)
| Task | Model | Metric | Baseline | Improvement |
|---|---|---|---|---|
| Forecasting | Prophet | WAPE | seasonal naive | |
| Churn | Gradient boosting | PR-AUC | logistic regression | |
| Segmentation | KMeans k=4 | silhouette | n/a | |

## Docker and Kubernetes
```bash
docker compose up --build      # api :8000, dashboard :8501, mlflow :5000
kubectl apply -f k8s/retailpulse.yaml   # see header comments for kind/minikube steps
```
Run the pipeline (`make all`) first so `data/processed/` exists.

## Roadmap
- [ ] Swap synthetic data for Online Retail II and record real results
- [ ] LSTM comparison (`pip install -r requirements-lstm.txt`, `python -m src.forecasting.train --lstm`)
- [ ] Orchestrate the pipeline with Prefect or Airflow
- [ ] Scheduled retraining on drift (GitHub Actions cron or a Kubernetes CronJob)
- [ ] Register models in the MLflow Model Registry and serve from it
- [ ] Evidently reports, Prometheus/Grafana monitoring
- [ ] Cloud deploy (GKE/EKS) with Ingress
