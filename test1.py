import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries
from skforecast.model_selection_multiseries import backtesting_forecaster_multiseries
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
import os

os.environ['OMP_NUM_THREADS'] = '1'

warnings.filterwarnings("ignore")


def split_time_data(df):
    end_train = '2024-04-15'
    data_train = df.loc[:end_train, :].copy()
    data_test = df.loc[end_train:, :].copy()
    print(f"Train dates      : {data_train.index.min()} --- {data_train.index.max()}  (n={len(data_train)})")
    print(f"Test dates       : {data_test.index.min()} --- {data_test.index.max()}  (n={len(data_test)})")

    return data_train, data_test


def percentage_error(actual, predicted):
    res = np.empty(actual.shape)
    for j in range(actual.shape[0]):
        if actual[j] != 0:
            res[j] = (actual[j] - predicted[j]) / actual[j]
        else:
            res[j] = predicted[j] / np.mean(actual)
    return res


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs(percentage_error(np.asarray(y_true), np.asarray(y_pred)))) * 100


df_train = pd.read_csv('train.csv', delimiter=";")

df_train['datetime'] = pd.to_datetime(df_train['DATE']+' '+df_train['TIME'])
df_train = df_train[df_train['TICKER'] == 'CHMF']
# print(df_train)
df_train = df_train.drop(['TICKER','TIME','DATE','PER'], axis=1)
df_train.set_index('datetime', inplace=True)
# print(df_train)

df_train_60T=df_train.resample('60T').sum()

# df_train.drop('VOL',axis=1).plot(figsize=(20,6))
# plt.show()

data_train, data_test = split_time_data(df_train_60T)

# df_test = pd.read_csv("sample_submission.csv", delimiter=",")
# df_test['datetime'] = pd.to_datetime(df_test['DATE']+' '+df_test['TIME'])
# df_test = df_test[df_test['TICKER'] == 'CHMF']
# df_test.set_index('datetime', inplace=True)
# df_test = df_test.sort_values(by="datetime")

# data_train = df_train_60T
# data_test = df_test

# print(df_train)
# print(df_test)

items = list(df_train[['OPEN','HIGH','LOW','CLOSE',]].columns)
forecaster_ms = ForecasterAutoregMultiSeries(
    regressor=HistGradientBoostingRegressor(),
    lags=24,
    transformer_series=StandardScaler()
)
forecaster_ms.fit(data_train[items])

predict=forecaster_ms.predict(steps=len(data_test))
mean_absolute_percentage_error(data_test['CLOSE'],predict['CLOSE'])

print(mean_absolute_percentage_error(data_test['CLOSE'],predict['CLOSE']))