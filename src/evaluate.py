from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.models import (
    make_monthly_series,
    train_test_split_time_series,
    fit_sarima,
    predict_sarima,
    fit_prophet,
    predict_prophet,
)

def mape(y_true, y_pred) -> float:
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    eps = 1e-9
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100)

def evaluate_model(name: str, y_true: pd.Series, y_pred: pd.Series) -> dict:
    mse = mean_squared_error(y_true, y_pred)   
    rmse = np.sqrt(mse)                        

    return {
        "model": name,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": rmse,
        "MAPE_%": mape(y_true, y_pred),
    }


def plot_test_predictions(train, test, sarima_pred, prophet_pred):
    plt.figure(figsize=(12, 5))
    plt.plot(train.index, train.values, label="Train")
    plt.plot(test.index, test.values, label="Test Actual")
    plt.plot(test.index, sarima_pred.values, label="SARIMA Pred")
    plt.plot(test.index, prophet_pred.values, label="Prophet Pred")
    plt.title("Test Set Predictions: SARIMA vs Prophet")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.show()

def plot_future_forecast(ts, future_pred, ci = None, title = "Future Forecast"):
    plt.figure(figsize=(12,5))
    plt.plot(ts.index, ts.values, label="Historical")
    plt.plot(future_pred.index, future_pred.values, label = "Forecast")

    if ci is not None:
        plt.fill_between(
            future_pred.index,
            ci.iloc[:, 0].values,
            ci.iloc[:, 1].values,
            alpha = 0.2,
            label = "Confidence Interval",
        )
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.show()

def main():
    ts_path = "data/processed/monthly_sales.csv"
    if not Path(ts_path).exists():
        raise FileNotFoundError(
            f"{ts_path} not found. Run: python -m src.prepare_data"
        )
    ts = make_monthly_series(ts_path)

    train, test = train_test_split_time_series(ts, train_ratio=0.8)
    sarima_fit = fit_sarima(train)
    sarima_test_fc = predict_sarima(sarima_fit, steps=len(test))
    sarima_pred = sarima_test_fc.y_pred

    prophet_model = fit_prophet(train)
    prophet_fc_all = predict_prophet(prophet_model, train, periods=len(test))
    prophet_pred = prophet_fc_all.y_pred.iloc[-len(test):]

    results = []
    results.append(evaluate_model("SARIMA", test, sarima_pred))
    results.append(evaluate_model("Prophet", test, prophet_pred))
    results_df = pd.DataFrame(results).sort_values("RMSE")
    print("\n=== Model Comparison (Lower is better) ===")
    print(results_df.to_string(index=False))

    plot_test_predictions(train, test, sarima_pred, prophet_pred)

    best_model = results_df.iloc[0]["model"]
    print(f"\nBest model based on RMSE: {best_model}")

    if best_model == "SARIMA":
        sarima_full = fit_sarima(ts) 
        fc_12 = predict_sarima(sarima_full, steps=12) 
        plot_future_forecast( 
            ts, 
            fc_12.y_pred, 
            ci=fc_12.conf_int, 
            title="SARIMA Forecast - Next 12 Months" 
        )
        print("\nNext 12 months forecast (SARIMA):") 
        print(fc_12.y_pred)
    else:
        prophet_full = fit_prophet(ts)
        fc_all = predict_prophet(prophet_full, ts, periods=12)
        future_12 = fc_all.y_pred.iloc[-12:] 
        future_ci_12 = fc_all.conf_int.iloc[-12:]
        plot_future_forecast( 
            ts, 
            future_12, 
            ci=future_ci_12, 
            title="Prophet Forecast - Next 12 Months" 
        )
        print("\nNext 12 months forecast (Prophet):") 
        print(future_12)

if __name__ == "__main__":
    main()