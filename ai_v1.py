import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np


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

# train_data = pd.read_csv("reduced_train.csv", delimiter=",")
train_data = pd.read_csv("train.csv", delimiter=";")
train_data['datetime'] = pd.to_datetime(train_data['DATE']+' '+train_data['TIME'])
train_data.set_index('datetime', inplace=True)

company_dict = {"VKCO":0,
                "SBER":0,
                "YNDX":0,
                "CHMF":0,
                "MTSS":0,
                "SMLT":0,
                "AGRO":0,
                "SIBN":0
                }
for item in company_dict:
    train_data_copy = train_data[train_data["TICKER"] == item]
    # train_data_copy["CLOSE"].plot()
    # plt.title(f"{item}")
    # plt.show()
    train = train_data_copy.iloc[:int(.99 * len(train_data_copy)), :]
    test = train_data_copy.iloc[int(.99 * len(train_data_copy)):, :]
    features = ["OPEN", "HIGH", "LOW", "VOL"]
    target = "CLOSE"

    model = xgb.XGBRegressor()
    model.fit(train[features], train[target])

    predictions = model.predict(test[features])
    # val = mean_absolute_percentage_error(train_test["CLOSE"], predictions)
    val = model.score(test[features], test[target])
    # print(val)
    plt.plot(train_data_copy["CLOSE"], label= "CLOSE PRICE")
    plt.plot(test[target].index, predictions, label="PREDICTIONS")
    plt.legend()
    plt.title(f"{item}")
    plt.show()
