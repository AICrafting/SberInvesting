# pip install streamlit prophet plotly
from datetime import datetime

import streamlit as st
import pandas as pd

from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

st.title('Stock Forecast')

stocks = ('VKCO', 'SBER', 'YNDX', 'CHMF', 'MTSS', 'SMLT', 'AGRO', 'SIBN')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

n_months = st.slider('Months of prediction:', 1, 4)
period = n_months * 31 * 24


@st.cache
def load_data(ticker):
    data = train_data[train_data["TICKER"] == ticker]
    data.reset_index(inplace=True)
    return data


train_data = pd.read_csv("train.csv", delimiter=";")
train_data['datetime'] = pd.to_datetime(train_data['DATE']+' '+train_data['TIME'])

data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')


st.subheader('Raw data')
st.write(data.tail())


# Plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['datetime'], y=data['OPEN'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['datetime'], y=data['CLOSE'], name="stock_close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


plot_raw_data()

# Predict forecast with Prophet.
df_train = data[['datetime', 'CLOSE']]
df_train = df_train.rename(columns={"datetime": "ds", "CLOSE": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period, freq="h")
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast.tail())

st.write(f'Forecast plot for {n_months} months')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)