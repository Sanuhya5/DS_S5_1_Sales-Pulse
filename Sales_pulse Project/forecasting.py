import warnings
import numpy as np
import pandas as pd

from evaluation import calculate_metrics, chronological_split


def make_daily_series(data):
    data = data[["date", "target"]].copy()

    data["date"] = pd.to_datetime(
        data["date"], errors="coerce"
    ).dt.normalize()

    data["target"] = pd.to_numeric(
        data["target"], errors="coerce"
    )

    data = data.dropna(subset=["date", "target"])

    data = data.groupby(
        "date", as_index=False
    )["target"].sum()

    data = data.sort_values("date")

    if data.empty:
        raise ValueError("No valid forecasting data.")

    dates = pd.date_range(
        data["date"].min(),
        data["date"].max(),
        freq="D"
    )

    data = data.set_index("date").reindex(dates)
    data.index.name = "date"

    # Internal missing dates are interpolated.
    # This is an assumption, not proof of zero or normal sales.
    data["target"] = data["target"].interpolate(
        method="time",
        limit_area="inside"
    )

    data = data.dropna(subset=["target"]).reset_index()

    return data


def baseline_forecast(training, future_dates):
    history = training["target"].to_numpy(dtype=float)

    predictions = []

    for future_date in pd.to_datetime(future_dates):
        days_ahead = (
            future_date - training["date"].max()
        ).days

        if len(history) >= 7:
            index = len(history) - 7 + (
                (days_ahead - 1) % 7
            )
            predictions.append(float(history[index]))
        else:
            predictions.append(float(history[-1]))

    return np.asarray(predictions)


def arima_forecast(training, future_dates):
    from statsmodels.tsa.arima.model import ARIMA

    candidate_orders = [
        (1, 1, 1),
        (2, 1, 1),
        (1, 1, 0),
        (1, 0, 1)
    ]

    best_model = None
    best_aic = float("inf")

    for order in candidate_orders:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                model = ARIMA(
                    training["target"].to_numpy(dtype=float),
                    order=order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                ).fit()

            if np.isfinite(model.aic) and model.aic < best_aic:
                best_model = model
                best_aic = model.aic

        except Exception:
            continue

    if best_model is None:
        raise ValueError("ARIMA could not fit this dataset.")

    return np.asarray(
        best_model.forecast(steps=len(future_dates)),
        dtype=float
    )


def prophet_forecast(training, future_dates):
    from prophet import Prophet

    model_data = training.rename(
        columns={"date": "ds", "target": "y"}
    )[["ds", "y"]]

    model = Prophet(
        weekly_seasonality=True,
        yearly_seasonality="auto",
        daily_seasonality=False
    )

    model.fit(model_data)

    future = pd.DataFrame({
        "ds": pd.to_datetime(future_dates)
    })

    predictions = model.predict(future)

    return predictions["yhat"].to_numpy(dtype=float)


def predict_with_model(name, training, future_dates):
    if name == "Baseline":
        return baseline_forecast(training, future_dates)

    if name == "ARIMA":
        return arima_forecast(training, future_dates)

    if name == "Prophet":
        return prophet_forecast(training, future_dates)

    raise ValueError("Unknown forecasting model.")


def run_forecast(data, horizon=30, model_choice="Compare all"):
    if horizon not in [7, 30, 60, 90]:
        raise ValueError("Choose a 7, 30, 60, or 90 day horizon.")

    series = make_daily_series(data)

    training, testing = chronological_split(series)

    if model_choice == "Baseline only":
        models = ["Baseline"]
    elif model_choice == "ARIMA only":
        models = ["ARIMA"]
    elif model_choice == "Prophet only":
        models = ["Prophet"]
    else:
        models = ["Baseline", "ARIMA", "Prophet"]

    metric_rows = []

    for model_name in models:
        try:
            predictions = predict_with_model(
                model_name,
                training,
                testing["date"]
            )

            metrics = calculate_metrics(
                testing["target"],
                predictions
            )

            metric_rows.append({
                "Model": model_name,
                **metrics,
                "Status": "Available"
            })

        except Exception as error:
            metric_rows.append({
                "Model": model_name,
                "MAE": np.nan,
                "RMSE": np.nan,
                "sMAPE (%)": np.nan,
                "Status": f"Unavailable: {error}"
            })

    metrics_table = pd.DataFrame(metric_rows)

    successful = metrics_table[
        metrics_table["Status"] == "Available"
    ].copy()

    if successful.empty:
        raise ValueError(
            "No forecasting model could produce a valid result."
        )

    best_model = successful.sort_values("MAE").iloc[0]["Model"]

    last_date = series["date"].max()

    future_dates = pd.date_range(
        last_date + pd.Timedelta(days=1),
        periods=horizon,
        freq="D"
    )

    predictions = predict_with_model(
        best_model,
        series,
        future_dates
    )

    # Display sales/units forecasts as non-negative.
    predictions = np.maximum(predictions, 0)

    forecast_table = pd.DataFrame({
        "date": future_dates,
        "prediction": predictions,
        "model": best_model
    })

    return {
        "history": series,
        "metrics": metrics_table,
        "forecast": forecast_table,
        "selected_model": best_model,
        "test": testing
    }
