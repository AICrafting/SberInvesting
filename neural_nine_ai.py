import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.layers.recurrent import LSTM

company = "VKCO"

end_train = '2024-05-01'

data = pd.read_csv("train.csv", delimiter=";")
data['datetime'] = pd.to_datetime(data['DATE']+' '+data['TIME'])
data.set_index('datetime', inplace=True)

CHMF_data = data[data["TICKER"] == company]

CHMF_train = CHMF_data.loc[:end_train, :].copy()
CHMF_test = CHMF_data.loc[end_train:, :].copy()

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(CHMF_train["CLOSE"].values.reshape(-1,1))

prediction_days = 30

x_train = []
y_train = []

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1)) #Prediction of next price

model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(x_train, y_train, epochs=3, batch_size=32)

test_prices = CHMF_test["CLOSE"].values

model_inputs = CHMF_data["CLOSE"][len(CHMF_data)-len(CHMF_test)-prediction_days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)#TODO look at how to reshape

# Make predictions

x_test = []
for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x-prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

plt.plot(CHMF_data["CLOSE"], color="black", label="Actual")
plt.plot(predicted_prices, color="green", label="Predicted")
plt.legend()
plt.show()