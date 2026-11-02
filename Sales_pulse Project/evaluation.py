
import numpy as np


def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    valid = np.isfinite(actual) & np.isfinite(predicted)

    actual = actual[valid]
    predicted = predicted[valid]

    if len(actual) == 0:
        return {
            "MAE": float("nan"),
            "RMSE": float("nan"),
            "sMAPE (%)": float("nan")
        }

    errors = actual - predicted

    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))

    denominator = np.abs(actual) + np.abs(predicted)
    valid_denominator = denominator > 0

    if valid_denominator.any():
        smape = np.mean(
            200 * np.abs(errors[valid_denominator])
            / denominator[valid_denominator]
        )
    else:
        smape = 0.0

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "sMAPE (%)": float(smape)
    }


def chronological_split(data):
    if len(data) < 20:
        raise ValueError(
            "At least 20 daily observations are recommended."
        )

    test_size = max(7, min(30, int(len(data) * 0.2)))
    test_size = min(test_size, len(data) // 3)

    training_data = data.iloc[:-test_size].copy()
    testing_data = data.iloc[-test_size:].copy()

    return training_data, testing_data
