import numpy as np


def seasonal_naive(train, horizon: int, season: int = 7) -> np.ndarray:
    """Repeat the last `season` days. The baseline every model must beat."""
    last = np.asarray(train)[-season:]
    return np.tile(last, int(np.ceil(horizon / season)))[:horizon].astype(float)


def wape(y_true, y_pred) -> float:
    """Weighted absolute percentage error. Safe when demand has zeros (unlike MAPE)."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    denom = np.abs(y_true).sum()
    return float(np.abs(y_true - y_pred).sum() / denom) if denom else float("nan")


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
