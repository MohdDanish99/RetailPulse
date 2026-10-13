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

Create and activate a Python virtual environment.

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the project dependencies according to the repository instructions, then run the available pipeline commands.

```bash
make setup
make all
make dashboard
make api
make mlflow
make test
```

Note: These commands require the corresponding Makefile targets and dependencies to be configured.

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