# pip install streamlit, prophet, plotly, pandas — загрузка модулей
import streamlit as st # Библиотека streamlit позволяет показать даннэ визуально в браузере, что позволяет сохранить время инвесторам
import pandas as pd # Pandas — главная библиотека в Python для работы с данными. (Позволяет взаимодействовать с файлами формата csv (читать данные в файлах))
from prophet import Prophet # Prophet — библиотека, позволяющая обучить итоговою модель, используя данные
from prophet.plot import plot_plotly
from plotly import graph_objs as go # Позволяет отображать графики в приложении

st.title('Stock Forecast') # Название приложения

stocks = ('VKCO', 'SBER', 'YNDX', 'CHMF', 'MTSS', 'SMLT', 'AGRO', 'SIBN') #tuple со всеми акциями в train.csv
selected_stock = st.selectbox('Select dataset for prediction', stocks) #checkbox на сайте, который позволяет выбрать нужную акцию

n_months = st.slider('Months of prediction:', 1, 4)
period = n_months * 31 * 24 # Сколько раз будет предпологаться цена акций - н месяцов * 31 день * 24 часа


@st.cache
def load_data(ticker):
    data = train_data[train_data["TICKER"] == ticker]
    data.reset_index(inplace=True)
    return data


train_data = pd.read_csv("train.csv", delimiter=";") # Чтение данных с train.csv
train_data['datetime'] = pd.to_datetime(train_data['DATE']+' '+train_data['TIME']) # Создание параметра datetime

data_load_state = st.text('Loading data...') # Визуальные эффекты на сайте
data = load_data(selected_stock) # Чтение данных с определенным значением TICKER
data_load_state.text('Loading data... done!') # Визуальные эффекты на сайте


st.subheader('Raw data') # Визуальные эффекты на сайте
st.write(data.tail()) # Показывает данные в таблице на сайте


def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['datetime'], y=data['OPEN'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['datetime'], y=data['CLOSE'], name="stock_close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


plot_raw_data() # Показывает график цены акции

df_train = data[['datetime', 'CLOSE']] # Используются только перемены datetime и CLOSE
df_train = df_train.rename(columns={"datetime": "ds", "CLOSE": "y"}) # Обязательная конструкция для модуля Prophet

m = Prophet() # ИИ Модель
m.fit(df_train) # Данные подаются в модель
future = m.make_future_dataframe(periods=period, freq="h") # Почасовая разметка на будующие н месяцов
forecast = m.predict(future) # Предсказание будущих значений CLOSE (dataframe в котором приситсвуют все данные df_train + предсказанные данные)

# Таблица предсказанных данных
st.subheader('Forecast data')
st.write(forecast.tail())
# График предсказанной цены
st.write(f'Forecast plot for {n_months} months')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)
# Графики - тренд по дням недели, тренд по времени дня
st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)