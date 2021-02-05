import math
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error


def symmetric_mean_absolute_percentage_error(actual, forecast):
    F = np.array(forecast)
    A = np.array(actual)
    return 100 / max(len(A), len(F)) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))


def naive_forecast_error(actual):
    A = np.array(actual)
    P = np.roll(A, 1)
    P[0] = A[0]
    return np.sum(np.abs(A - P))


def mean_absolute_scaled_error(actual, forecast):
    F = np.array(forecast)
    A = np.array(actual)
    return np.sum(np.abs(F - A)) / naive_forecast_error(actual)


def interval_accuracy_score(actual, forecast):
    conf = 0.9
    z = 1.645
    A = np.array(actual)
    F = np.array(forecast)
    E = np.abs(A - F)
    std_dev = np.std(E)
    mean = np.mean(np.abs(E))
    lb = mean - z * std_dev / math.sqrt(len(forecast))
    ub = mean + z * std_dev / math.sqrt(len(forecast))
    print(f'{std_dev},{mean},{lb},{ub}')
    return np.sum(ub - lb + np.where(E < lb, 1, 0) * (lb - E) * 2 / conf +
                  np.where(E > ub, 1, 0) * (E - ub) * 2 / conf) / len(E)


def eval_model(predictions, pred_step, actual):
    predictions = predictions[::pred_step]
    actual = actual[::pred_step]

    print("Total Values: {}".format(min(len(predictions), len(actual))))
    print("Forecast: {}".format(predictions))
    print("Validation Set: {}".format(actual))
    print()

    mae = mean_absolute_error(actual, predictions)
    rmse = math.sqrt(mean_squared_error(actual, predictions))
    ignore_zero_values = np.where(np.array(actual) == 0, 0, 1)
    mape = mean_absolute_percentage_error(actual, predictions, ignore_zero_values)
    smape = symmetric_mean_absolute_percentage_error(actual, predictions)
    mase = mean_absolute_scaled_error(actual, predictions)
    ias = interval_accuracy_score(actual, predictions)

    print("Mean Absolute Error: {}".format(mae))
    print("Root Mean Squared Error: {}".format(rmse))
    print("Mean Absolute Percentage Error: {}".format(mape))
    print("Symmetric Mean Absolute Percentage Error: {}".format(smape))
    print("Mean Absolute Scaled Error: {}".format(mase))
    print("Mean Interval Accuracy Score: {}".format(ias))


if __name__ == '__main__':
    forecast = [5, 3, 8, 2, 1, 1, 7, 5, 5]
    actual = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    eval_model(actual, 1, actual)
