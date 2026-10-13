"""Optional LSTM forecaster. Needs: pip install -r requirements-lstm.txt"""
import numpy as np
import pandas as pd


def lstm_forecast(series: pd.Series, horizon: int, window: int = 28, epochs: int = 30) -> np.ndarray:
    from tensorflow import keras

    values = series.values.astype("float32")
    scale = max(values.max(), 1.0)
    v = values / scale
    X = np.array([v[i : i + window] for i in range(len(v) - window)])[..., None]
    y = v[window:]
    model = keras.Sequential(
        [keras.layers.Input((window, 1)), keras.layers.LSTM(32), keras.layers.Dense(1)]
    )
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=epochs, batch_size=32, verbose=0)

    hist, preds = list(v[-window:]), []
    for _ in range(horizon):  # recursive multi-step forecast
        nxt = float(model.predict(np.array(hist[-window:])[None, :, None], verbose=0)[0, 0])
        preds.append(nxt)
        hist.append(nxt)
    return np.clip(np.array(preds) * scale, 0, None)
