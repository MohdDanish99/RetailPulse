.PHONY: setup data etl forecast segment churn inventory drift all api dashboard mlflow test lint up

setup:
	pip install -r requirements.txt

data:
	python -m src.etl.generate_sample_data

etl:
	python -m src.etl.pipeline

forecast:
	python -m src.forecasting.train

segment:
	python -m src.segmentation.rfm

churn:
	python -m src.churn.train

inventory:
	python -m src.inventory.optimizer

drift:
	python -m src.monitoring.drift

all: data etl forecast segment churn inventory drift

api:
	uvicorn src.api.main:app --reload --port 8000

dashboard:
	streamlit run dashboard/app.py

mlflow:
	mlflow ui --port 5000

test:
	pytest

lint:
	ruff check .

up:
	docker compose up --build
