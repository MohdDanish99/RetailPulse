# RetailPulse: AI-Powered Customer Analytics & Demand Forecasting

RetailPulse is an AI-powered customer analytics and demand forecasting platform that analyzes sales, customer, and inventory data to help retail businesses make smarter decisions.

The platform provides demand forecasting, customer segmentation, churn prediction, and inventory optimization through machine learning, APIs, dashboards, and MLOps tools.

## Architecture

```text
Raw CSV Data
    |
    v
ETL Pipeline (Cleaning, Deduplication, Aggregation)
    |
    v
Processed Data (Parquet)
    |
    +-- Demand Forecasting
    +-- Customer Segmentation (RFM + KMeans)
    +-- Churn Prediction
    +-- Inventory Optimization
    |
    v
MLflow Tracking
    |
    v
FastAPI + Streamlit Dashboard
    |
    v
Docker and Kubernetes
```

## Quick Start

The fastest way to run the dashboard and API is with Docker Desktop. The Compose setup reads the processed files in `data/processed`; it does not run the analytics pipeline automatically.

```powershell
docker compose build
docker compose up -d
docker compose ps
```

Open these local URLs:

| Service | URL |
|---|---|
| Streamlit dashboard | <http://localhost:8501> |
| FastAPI | <http://localhost:8000> |
| API documentation | <http://localhost:8000/docs> |
| MLflow (optional) | <http://localhost:5000> |

The Compose ports bind to `127.0.0.1`, so these services are available from the same computer. MLflow is not exposed through the public demo tunnel.

### Generate or refresh demo data

To generate synthetic sample data and run all analytics stages, use a Python environment with the project dependencies installed and GNU Make available (for example, in Ubuntu on WSL):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
make all
```

**Warning:** `make all` starts by generating synthetic input data and overwrites `data/raw/online_retail_II.csv` and `data/raw/inventory.csv`. Do not run it if those files contain real data you want to preserve. For real data, run the required Makefile stages deliberately, starting with `make etl`, and run forecasting before inventory optimization. See [deployement_report.md](deployement_report.md) for input columns and detailed deployment guidance.

## No-cost public demo (temporary)

You can share the locally running application over the internet without a paid server or domain using a Cloudflare Quick Tunnel. This is intended for demos and testing, not as a permanent production host. Your computer and Docker Desktop must remain on, and the generated public URL changes when the tunnel is restarted. Anyone who has the public URL can access the demo unless you use the protected option below.

Start the tunnel in PowerShell from the repository root:

```powershell
docker compose -f docker-compose.yml -f docker-compose.tunnel.yml up -d
docker compose -f docker-compose.yml -f docker-compose.tunnel.yml logs -f cloudflared
```

Copy the `https://....trycloudflare.com` URL printed in the tunnel logs. The root URL serves the dashboard. The API is available on the same host, for example `/docs`, `/health`, `/forecast/P036`, `/segment/12000`, `/churn/12001`, and `/inventory/P000` (use IDs and SKUs present in your own outputs).

To require visitors to verify an allowed email address with a one-time PIN, run the protected variant instead:

```powershell
$env:ALLOWED_MAIL = "you@example.com"
docker compose -f docker-compose.yml -f docker-compose.tunnel.yml -f docker-compose.tunnel.protected.yml up -d
docker compose -f docker-compose.yml -f docker-compose.tunnel.yml -f docker-compose.tunnel.protected.yml logs -f cloudflared
```

Replace the example email with the address that should be allowed. Cloudflare's email-protected Quick Tunnel feature does not require a Cloudflare account or custom domain. See the [Quick Tunnel documentation](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/trycloudflare/) and [email protection instructions](https://developers.cloudflare.com/changelog/post/2026-10-02-protected-quick-tunnels/).

Stop the public tunnel and local services. For an unprotected tunnel:

```powershell
docker compose -f docker-compose.yml -f docker-compose.tunnel.yml down
```

If you started the protected variant, set the allowed email again and include its Compose file:

```powershell
$env:ALLOWED_MAIL = "you@example.com"
docker compose -f docker-compose.yml -f docker-compose.tunnel.yml -f docker-compose.tunnel.protected.yml down
```

A Quick Tunnel has no uptime guarantee and should be used only for testing or demos.

## Main Features

- **Demand Forecasting:** Predict future sales and compare forecasting models against a seasonal-naive baseline.
- **Customer Segmentation:** Group customers using Recency, Frequency, and Monetary (RFM) analysis and KMeans.
- **Churn Prediction:** Identify customers who may stop purchasing.
- **Inventory Optimization:** Calculate safety stock, reorder points, and economic order quantities.
- **Model Monitoring:** Track model performance and detect data drift using the Population Stability Index (PSI).
- **API and Dashboard:** Provide model outputs through FastAPI and Streamlit.
- **MLOps:** Track experiments with MLflow and support containerization with Docker.

## Dataset

The project can use synthetic retail data for initial testing or the Online Retail II dataset for real-world analysis.

For real data, prepare the dataset according to the ETL pipeline's expected columns and configuration. Inventory optimization also requires inventory information such as stock levels, supplier lead times, and unit costs.

## Evaluation

Forecasting models can be evaluated using WAPE and RMSE. Churn models can be evaluated using PR-AUC and recall. Customer segmentation can be evaluated using silhouette scores.

Record actual evaluation results after running the pipeline. Model performance depends on the dataset and evaluation setup.

## Docker and Kubernetes

The project can be containerized with Docker Compose and deployed to Kubernetes using the provided configuration files, once the required services and configurations are ready.

## Roadmap

- [ ] Evaluate models on real retail data.
- [ ] Compare LSTM forecasting with baseline models.
- [ ] Add workflow orchestration using Prefect or Airflow.
- [ ] Automate model retraining.
- [ ] Add model registry and monitoring dashboards.
- [ ] Deploy to a cloud environment.

## Project Status

This project is intended as an end-to-end retail analytics platform. Features and deployment capabilities should be verified by running the implementation and its tests.
