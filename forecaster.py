import warnings

import keras
import pandas
import mchmm as mc
from pandas import DataFrame
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as pp
import statsmodels.api as sm
import accuracy
import datetime
m
import itertools


def dateparse(time_in_secs):
    return datetime.datetime.fromtimestamp(float(time_in_secs))


def read_data(filepath):
    f = open(filepath, 'r')
    df = pandas.read_csv(f, delimiter=', ', engine='python', parse_dates=True,
                         date_parser=dateparse, index_col='timestamp')
    # with pandas.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    # print(df)
    return df


def plot(filepath, predictions, title, trainingdata=None):
    pp.plot(predictions, label='Prediction')
    if trainingdata is not None:
        pp.plot(trainingdata, color='red', label='Training Data')
        pp.legend(loc="upper left")
    pp.ylabel('count')
    pp.xlabel('timestep')
    pp.title(title)
    pp.savefig(filepath, dpi=300)
    pp.clf()


def forecast_markov(trainingdata, evaldata):
    n = len(evaldata)
    mod = mc.MarkovChain().from_data(trainingdata['count'])
    print(evaldata['count'][0])
    ids, states = mod.simulate(n, start=10)
    plot(f'markov_{n}.png', states, 'Forecast based on Markov Chains',
         evaldata['count'].values)
    print('Accuracy Scores for Markov Chains:')
    accuracy.eval_model(states, 1, evaldata['count'].values)


def forecast_linear_reg(trainingdata, evaldata):
    n = len(evaldata)
    to_timestamp_converter = lambda t: (t - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
    x = np.array([to_timestamp_converter(t) for t in trainingdata.index.values]).reshape((-1, 1))
    print(x)
    y = np.array(trainingdata['count'])
    model = LinearRegression().fit(x, y)
    predictions = model.predict(x[:n])
    plot(f'linear_reg_{n}.png', predictions,
         'Forecast based on Linear Regression (OLS)',
         evaldata['count'].values)
    print('Summary')
    print(f'Coefficient of Determination: {model.score(x, y)}')
    print(f'Function: {model.coef_}t + {model.intercept_}')
    print('Accuracy Scores for OLS:')
    accuracy.eval_model(predictions, 1, evaldata['count'].values)


def forecast_sarimax(trainingdata, evaldata):
    n = len(evaldata)
    periods_in_season = 24

    mod = sm.tsa.statespace.SARIMAX(trainingdata['count'].astype(float), order=(0, 0, 0),
                                    seasonal_order=(1, 0, 1, periods_in_season),
                                    enforce_stationarity=False)
    res = mod.fit(disp=False)
    print(res.summary())

    predictions = res.forecast(n)
    print(predictions)
    accuracy.eval_model(predictions.values, 1, evaldata['count'].values)
    plot(f'sarimax/sarimax_s{periods_in_season}_{n}.png', predictions.values,
         'Forecast based on the SARIMAX Model', evaldata['count'].values)



def forecast_neural(trainingdata, evaldata):
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(np.array(trainingdata['count'].astype(float)).reshape(-1, 1))
    validationset = scaler.fit_transform(np.array(evaldata['count'].astype(float)).reshape(-1, 1))
    look_back = 20
    trainX, trainY = create_dataset(dataset, look_back)
    testX, testY = create_dataset(validationset, look_back)
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

    try:
        model = keras.models.load_model(f"lstm_network_{look_back}")
        print('Successfully loaded model.')
    except IOError:
        print('Creating new model.')
        model = Sequential()
        model.add(LSTM(4, input_shape=(1, look_back)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
        model.save(f'lstm_network_{look_back}')


    # make predictions
    # trainPredict = model.predict(trainX)
    # testPredict = model.predict(testX)
    prediction_list = trainX[-1]

    for _ in range(len(evaldata)):
        x = prediction_list[-look_back:]
        x = x.reshape((1, 1, look_back))
        out = model.predict(x)[0][0]
        prediction_list = np.append(prediction_list, out)
    prediction_list = prediction_list[look_back:]
    # invert predictions
    print(np.shape(testX))
    print(testX)
    testPredict = scaler.inverse_transform(prediction_list.reshape(-1, 1))
    testX = scaler.inverse_transform(testX)
    accuracy.eval_model(testPredict[:,0], 1, evaldata['count'].values)
    plot(f'lstm_l{look_back}_{len(testPredict[:,0])}.png', testPredict[:,0],
         'Forecast based on a LSTM Neural Network', evaldata['count'].values)


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


def create_sequences(dataset):
    training_length = 50
    features = []
    for i in range(training_length, len(dataset)):
        extract = dataset[i - training_length:i + 1]
        features.append(extract[:-1])
    features = np.array(features)
    return features

data = read_data('KafkaTrainingData.txt')
train = data.iloc[:1000]
validation = data.iloc[1000:1100]
# plot('trainingdata100.png', trainingdata['count'].head(100))
# forecast_markov(train, validation)
# forecast_linear_reg(train, validation)
# forecast_sarimax(train, validation)
forecast_neural(train, validation)
# forecast_machine(train, validation)
