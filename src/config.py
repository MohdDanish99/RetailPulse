from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"

RAW_FILE = DATA_RAW / "online_retail_II.csv"
INVENTORY_FILE = DATA_RAW / "inventory.csv"

CHURN_DAYS = 90        # churn = no purchase in the next 90 days
FORECAST_HORIZON = 28  # days
TOP_N_SKUS = 20        # SKUs forecast and optimised
