import json
import math

import paho.mqtt.client as mqtt
from tensorflow import keras

import pickle
from datetime import datetime
from time import time

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler


topic = '25_76'
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
    return np.sum(ub - lb + np.where(E < lb, 1, 0) * (lb - E) * 2 / conf +
                  np.where(E > ub, 1, 0) * (E - ub) * 2 / conf) / len(E)


def eval_model(predictions, pred_step, actual, verbose=False):
    predictions = predictions[::pred_step]
    actual = actual[::pred_step]

    mae = mean_absolute_error(actual, predictions)
    rmse = math.sqrt(mean_squared_error(actual, predictions))
    ignore_zero_values = np.where(np.array(actual) == 0, 0, 1)
    mape = mean_absolute_percentage_error(actual, predictions, ignore_zero_values)
    smape = symmetric_mean_absolute_percentage_error(actual, predictions)
    mase = mean_absolute_scaled_error(actual, predictions)
    ias = interval_accuracy_score(actual, predictions)
    if verbose:
        print("Total Values: {}".format(min(len(predictions), len(actual))))
        print("Forecast: {}".format(predictions))
        print("Validation Set: {}".format(actual))
        print()
        print("Mean Absolute Error: {}".format(mae))
        print("Root Mean Squared Error: {}".format(rmse))
        print("Mean Absolute Percentage Error: {}".format(mape))
        print("Symmetric Mean Absolute Percentage Error: {}".format(smape))
        print("Mean Absolute Scaled Error: {}".format(mase))
        print("Mean Interval Accuracy Score: {}".format(ias))
    metrics = dict(mae=mae, rmse=rmse, mape=mape, smape=smape, mase=mase, ias=ias)
    return metrics


def eval_markov(data, mqttc):
   mod = pickle.load(open('models/markovmodel.fdml', 'rb'))
   ids, predictions = mod.simulate(len(data)-1, start=int(data['count'][0]))
   metrics = eval_model(predictions, 1, data['count'][1:])
   send_metrics('markov', metrics, mqttc)


def eval_linreg(data, mqttc):
    mod = pickle.load(open('models/linregmodel.fdml', 'rb'))
    to_timestamp_converter = lambda t: (t - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
    x = np.array([to_timestamp_converter(t) for t in data.index.values]).reshape((-1, 1))
    predictions = mod.predict(x)
    metrics = eval_model(predictions, 1, data['count'])
    print(metrics)
    send_metrics('linreg', metrics, mqttc)

def eval_arima(data, mqttc):
    mod = pickle.load(open('models/arimamodel.fdml', 'rb'))
    predictions = mod.forecast(len(data))
    metrics = eval_model(predictions, 1, data['count'])
    print(metrics)
    send_metrics('arima', metrics, mqttc)


def eval_rnn(data, mqttc):
    model = keras.models.load_model("models/rnnmodel")
    scaler = MinMaxScaler(feature_range=(0, 1))
    last_20 = scaler.fit_transform(np.array(data['count'].astype(float)).reshape(-1, 1))
    prediction_list = last_20[-20:]
    for _ in range(len(data) - 20):
        x = prediction_list[-20:]
        x = x.reshape((1, 1, 20))
        out = model.predict(x)[0][0]
        prediction_list = np.append(prediction_list, out)
    prediction_list = prediction_list[20:]
    # invert predictions
    testPredict = scaler.inverse_transform(prediction_list.reshape(-1, 1))
    metrics = eval_model(testPredict[:,0], 1, data['count'][20:])
    print(metrics)
    send_metrics('rnn', metrics, mqttc)


def send_metrics(modelname, metrics: dict, mqttc):
    timestamp = str(int(time()) * 1000)
    for key, value in metrics.items():
        message = {'username': 'group4_2020_ws',
                   f'{modelname}_{key}': str(value),
                   'device_id': "76",
                   'timestamp': timestamp}
        print(message)
        mqttc.publish(topic, json.dumps(message))


if __name__ == '__main__':
    CONSUMER_URL = 'https://iotplatform.caps.in.tum.de'
    DEV_JWT = 'eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJpYXQiOjE2MTI0NDk3NzQsImlzcyI6ImlvdHBsYXRmb3JtIiwic3ViIjoiMjVfNjMifQ.vgNjk3NN5xvDAKIvrZPClLg-wscqORVrgeB7cHYbzY4VUnPlYGILHWlPpY44w5A1GyyC_6vPWtQU1c9fAdnkbCOv9KnU2_dOUjB9InUzZRhVn8hGtT9K1oojszlO4gQfVa2hT8CiAClcYDTsNnBetqgb95-k6jepR8iEnd4EVQAvjDwDnI_-VD9cAZJW9kRLF-49zJrCX4vUPrNNQr9Qt2CD35HM0Dq1Gb272tDEz7lgTwe1U1xHx0VA9Fdjn6Hp0XSzePv4Le4z_-FivP1Dr1sFDf4U7jgrWbqUrE_oznl2Hlc9udGx_vFuSRXiiGb39DsBHDqBCeuMP4WkOKE7RQWyPx9UHvJgqWTC42tBPo0oc_KPfwgcE6YUXXYjX_ZMiblGqU0XPXa8mSkhKNxMEnMpoKmNNFjygQeLkuZ5eT6Dbc8S-KmrJ4BzKZUqg9zslzJIQcweLXsJpL9-y63NivdtT-zzR5PL3ZORc-Ok9DoZnFYvg7yzYx6PXydIWEMYpuTx1o0107C91K5Kf1Mli_Hv2YxQzulsAOmsff9CUSYfpxiRzTEsC7G3lHmRYTzi6k8LZRnFJKrJ_2l2PUUs2UYhyqezSQsfP_LmG4zg4iTsOQxb-GUYENeB7W5sssU3UJZVhvoAQWh7t8sjvtmQql2lx41eVc5lwM8RAHhGSvw'

    sensorID = str(856)
    timeHorizonWeeks = 0
    batchSize = 50
    searchPath = '/api/consumers/consume/' + sensorID + '/_search?'  # base path - search all
    countPath = '/api/consumers/consume/' + sensorID + '/_count?'  # base path - count all

    retrievedData = pd.DataFrame(columns=['t', 'count'])

    # I: https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-range-query.html
    if timeHorizonWeeks != 0:
        ts = time()
        curTs = int(ts - (ts % 60))
        weekAgoTs = int(curTs - timeHorizonWeeks * 7 * 24 * 60 * 60)
        sQuery = 'q=timestamp:[' + str(weekAgoTs * 1000) + '%20TO%20' + str(curTs * 1000) + ']'
        searchPath = searchPath + sQuery
        countPath = countPath + sQuery

    headers = {"Content-Type": "application/x-www-form-urlencoded", "Authorization": "Bearer " + DEV_JWT}
    # Get the data for the last week
    response = requests.get(CONSUMER_URL + countPath, headers=headers, verify=False)
    respCountData = response.json()
    countLeft = int(respCountData["count"])
    begIndex = 0

    while countLeft > 0:
        # Slicing our requests
        searchPath = searchPath + '&from=' + str(begIndex) + '&size=' + str(batchSize)

        response = requests.get(CONSUMER_URL + searchPath, headers=headers, verify=False)
        # I: https://docs.python.org/3/library/http.client.html#httpresponse-objects
        respData = response.json()
        observationsArray = respData["hits"]["hits"]

        for observation in observationsArray:
            # Important to convert to seconds, datetime cannot handle ms
            # TODO: consider rounding to minutes
            timestamp_s = int(observation['_source']['timestamp'] / 1000)
            count = int(observation['_source']['value'])
            cur_date = datetime.fromtimestamp(timestamp_s)
            df_row = pd.DataFrame([[cur_date, count]], columns=['t', 'count'])
            retrievedData = retrievedData.append(df_row)

        begIndex = begIndex + batchSize
        countLeft = countLeft - batchSize

    print(retrievedData)
    retrievedData.index = retrievedData.t

    username = 'JWT'
    password = 'eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJpYXQiOjE2MTI1NjI2ODMsImlzcyI6ImlvdHBsYXRmb3JtIiwic3ViIjoiMjVfNzYifQ.KZEuKMyLCtRIJ04CsvLTkEkpcZL2oCb64O0KpTrLrUwEBSXu4pmg0Zg_ju2ZFzkVgQgB1bCctqMVlyspkAgVNcOIgQ617I6wvtIuwa25nrSzozvtRzPuV-iYzaaKA0VtqhA_5jur9bA49Eb0oBNGKT7NJuO28Vr6h4B3zQ4QBZyFsW-fvgJKHpdhXTxxDkI25bPE6cdWhtbkylwU_NU_i59khTHdvYG09dik-35U107UBXQKz85dpQ1zRZVt3YBrjkEX7aUm5F4HSlcQY---Et_PN5CTXbYI3fQJCtYozXAVq8Ytn4f1VGQTESs5WDSxtAsklkeqPlO9_wtnva3M-6hiSg9-CPsNKZiZWJacXdjOjzfW6O-ejDFbCN0PqjFxFzRnmaKY81Kq2MRRLXRGi6WxozR5Gr_8PHqNi5PiS-q0L7gyXi0SIGmvDxYwo1NjOdoheTL9Et07KSCljlAEByPCH_JGkABs3ggfMX-Kvxr9ocsoaOTEQ6YlUHa6rYMFp8ImRp0sxBGR9Tgi-vjZVsH1O_bMWns6F8VxqmJ6OwmdSHCaYJbBsIdk8ginccMWcqhYNxBUPegWPkTKeW8huYuMsU7VgV6TXkEK7F8yZHOgGF9Yz6rhGyp7jgL87kjbWykwq0aWvrKSQAyJUvRIZvjld-QAWSX2S8K2EWji05g'
    address = '131.159.35.132'
    port = '1883'

    mqttc = mqtt.Client('', protocol=mqtt.MQTTv311)
    # mqttc.enable_logger(root)

    mqttc.username_pw_set(username, password)
    mqttc.connect(address)
    # eval_markov(retrievedData, mqttc)
    eval_linreg(retrievedData, mqttc)
    eval_arima(retrievedData, mqttc)
    eval_rnn(retrievedData, mqttc)
