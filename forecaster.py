import pandas
import mchmm as mc
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as pp
import statsmodels.api as sm
import accuracy
import datetime


def dateparse (time_in_secs):
    return datetime.datetime.fromtimestamp(float(time_in_secs))


def read_data(filepath):
    f = open(filepath, 'r')
    df = pandas.read_csv(f, delimiter=', ', engine='python', parse_dates=True,
                         date_parser=dateparse, index_col='timestamp')
    return df


def plot(filepath, predictions, title, trainingdata=None):
    pp.plot(predictions, label='Prediction')
    if trainingdata is not None:
        pp.plot(trainingdata['count'].values, color='red', label='Training Data')
        pp.legend(loc="upper left")
    pp.ylabel('count')
    pp.xlabel('timestep')
    pp.title(title)
    pp.savefig(filepath, dpi=300)
    pp.clf()


def forecast_markov(n, trainingdata):
    mod = mc.MarkovChain().from_data(trainingdata['count'])
    ids, states = mod.simulate(n, start=0)
    plot(f'markov_{n}.png', states, 'Forecast based on Markov Chains',
         trainingdata.head(n))
    print('Accuracy Scores for Markov Chains:')
    accuracy.eval_model(states, 1, trainingdata['count'].head(n).values)


def forecast_linear_reg(n, trainingdata):
    x = np.array(trainingdata['timestamp']).reshape((-1,1))
    y = np.array(trainingdata['count'])
    model = LinearRegression().fit(x, y)
    predictions = model.predict(x[:n])
    plot(f'linear_reg_{n}.png', predictions,
         'Forecast based on Linear Regression (OLS)', trainingdata.head(n))
    print('Summary')
    print(f'Coefficient of Determination: {model.score(x,y)}')
    print(f'Function: {model.coef_}t + {model.intercept_}')
    print('Accuracy Scores for OLS:')
    accuracy.eval_model(predictions, 1, trainingdata['count'].head(n).values)


def forecast_sarimax(n, trainingdata):
    mod = sm.tsa.statespace.SARIMAX(trainingdata['count'].astype(float), order=(0, 1, 0), seasonal_order=(1,0,0,12),
                                    enforce_stationarity=False)
    res = mod.fit(disp=False)
    print(res.summary())
    predictions = res.forecast(n).values
    print(type(predictions))
    accuracy.eval_model(predictions, 1, trainingdata['count'].head(n).values)
    plot(f'sarimax_{n}.png', predictions,
         'Forecast based on the SARIMAX Model', trainingdata.head(n))

def remove_seasonality(trainingdata):
    resample = trainingdata.resample('M')
    monthly_mean = resample.mean()
    print(monthly_mean.head(13))
    monthly_mean.plot()
    pp.savefig('no_seasonality')

data = read_data('KafkaTrainingData.txt')
# plot('trainingdata100.png', trainingdata['count'].head(100))
# forecast_markov(50, data)
# forecast_linear_reg(50, data)

forecast_sarimax(100, data)
