import numpy as np
import pandas as pd


def prophet_forecast(series: pd.Series, horizon: int) -> np.ndarray:
    from prophet import Prophet  # imported lazily so the rest of the repo works without it

    df = series.reset_index()
    df.columns = ["ds", "y"]
    model = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False)
    model.fit(df)
    future = model.make_future_dataframe(periods=horizon)
    return np.clip(model.predict(future)["yhat"].tail(horizon).values, 0, None)
