from dataclasses import dataclass
import pandas as pd
import numpy as np


from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet

@dataclass
class ForecastResult:
    y_pred: pd.Series
    conf_int: pd.DataFrame | None = None

def train_test_split_time_series(ts: pd.Series, train_ratio: float = 0.8):
    split_idx = int(len(ts) * train_ratio)
    train = ts.iloc[:split_idx]
    test = ts.iloc[split_idx:]
    return train, test

def fit_sarima(train: pd.Series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
    model = SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False)
    return fitted

def predict_sarima(fitted_model, steps: int) -> ForecastResult:
    fc=fitted_model.get_forecast(steps=steps)
    y_pred = fc.predicted_mean
    conf_int = fc.conf_int()
    return ForecastResult(y_pred=y_pred, conf_int=conf_int)

def fit_prophet(train: pd.Series) -> Prophet:
    df = train.reset_index()
    df.columns = ["ds", "y"]

    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
    )
    m.fit(df)
    return m

def predict_prophet(model: Prophet, history_ts: pd.Series, periods: int) -> ForecastResult:
    future = model.make_future_dataframe(periods=periods, freq="MS")
    fc = model.predict(future).set_index("ds")
    y_pred = fc["yhat"]
    conf_int = fc[["yhat_lower", "yhat_upper"]]
    return ForecastResult(y_pred=y_pred, conf_int=conf_int)

def make_monthly_series(monthly_sales_csv: str) -> pd.Series:
    df = pd.read_csv(monthly_sales_csv)
    df["date"] = pd.to_datetime(df["date"])
    ts = df.set_index("date")["sales"].sort_index()
    ts = ts.asfreq("MS")
    return ts