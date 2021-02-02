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
    return np.sum(np.abs(A - P)) / (len(A) - 1)


def mean_absolute_scaled_error(actual, forecast):
    F = np.array(forecast)
    A = np.array(actual)
    return np.sum(np.abs(F - A) / naive_forecast_error(actual)) / max(len(A), len(F))


def interval_accuracy_score(actual, lbs, ubs, conf):
    A = np.array(actual)
    L = np.array(lbs)
    U = np.array(ubs)
    return np.sum(U - L + np.where(A < L, 1, 0) * (L - A) * 2 / conf +
                  np.where(A > U, 1, 0) * (A - U) * 2 / conf) / len(A)


def eval_model(model, pred_step, counts):
    model = model[::pred_step]
    counts = counts[::pred_step]

    lbs = [5, 5, 5, 5, 5, 5, 5, 5, 5]
    ubs = [5, 5, 5, 5, 5, 5, 5, 5, 5]
    conf = 0.95

    print("Total Values: {}".format(min(len(model), len(counts))))
    print("Forecast: {}".format(model))
    print("Validation Set: {}".format(counts))
    print()

    mae = mean_absolute_error(model, counts)
    rmse = math.sqrt(mean_squared_error(model, counts))
    mape = mean_absolute_percentage_error(model, counts)
    smape = symmetric_mean_absolute_percentage_error(model, counts)
    mase = mean_absolute_scaled_error(model, counts)
    # ias = interval_accuracy_score(counts, lbs, ubs, conf)

    print("Mean Absolute Error: {}".format(mae))
    print("Root Mean Squared Error: {}".format(rmse))
    print("Mean Absolute Percentage Error: {}".format(mape))
    print("Symmetric Mean Absolute Percentage Error: {}".format(smape))
    print("Mean Absolute Scaled Error: {}".format(mase))
    # print("Mean Interval Accuracy Score: {}".format(ias))


if __name__ == '__main__':
    model = [5, 3, 8, 2, 1, 1, 7, 5, 5]
    counts = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    eval_model(model, 1, counts)
