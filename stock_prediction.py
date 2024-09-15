# ВНИМАМИЕ!!! - В этом архиве изменен файл train.csv - добавлено 8 строчек для правильной работы программы. (Поскольку архив с данными заканчивается на времени 23:49:00, а предсказания почасовые, в инном случае все предсказания имели бы форму TIME=XX:49:00). В остальном dataset не изменен. Просьба использовать предоставленный файл train.csv

# pip install prophet, pandas, csv — загрузка модулей
import csv # Позволяет взаимодействовать с файлами формата csv (писать данные в файлы)
import pandas as pd # Pandas — главная библиотека в Python для работы с данными. (Позволяет взаимодействовать с файлами формата csv (читать данные в файлах))
from prophet import Prophet # Prophet — библиотека, позволяющая обучить итоговою модель, используя данные

stocks = ('VKCO', 'SBER', 'YNDX', 'CHMF', 'MTSS', 'SMLT', 'AGRO', 'SIBN') #tuple со всеми акциями в train.csv

period = 31 * 24 # Сколько раз будет предпологаться цена акций - 31 день * 24 часа
# Возможная доработка - уменьшить количество предположений - цена не меняется когда рынок закрыт (К сожалению трудно выполнимо с модулем Prophet - смтр. стр. 35)


def load_data(ticker):
    data = train_data[train_data["TICKER"] == ticker]
    data.reset_index(inplace=True)
    return data


train_data = pd.read_csv("train.csv", delimiter=";") # Чтение данных с train.csv
train_data['datetime'] = pd.to_datetime(train_data['DATE']+' '+train_data['TIME']) # Создание параметра datetime

data_final = [["KEY", "TICKER", "DATE", "TIME", "CLOSE"]] # Будущие заголовки файла stock_prediction.csv
non_assessable_days = [10, 11 ,12 ,18 ,19, 25, 26, 1, 2, 8, 9] # Дни в которые биржа не работает в мае (10 июня учтено отдельно)
# Возможная доработка - успользовать модуль datetime, чтобы определить день недели - в таком случае придется вносить в этот список только 9 мая

for ticker in stocks: # Операция выполняется для каждого значения TICKER
    data = load_data(ticker) # Чтение данных с определенным значением TICKER

    # Предсказание будущих цен используя модуль Prophet
    df_train = data[['datetime', 'CLOSE']] # Используются только перемены datetime и CLOSE
    print(df_train)
    df_train = df_train.rename(columns={"datetime": "ds", "CLOSE": "y"}) # Обязательная конструкция для модуля Prophet

    m = Prophet() # ИИ Модель
    m.fit(df_train) # Данные подаются в модель
    future = m.make_future_dataframe(periods=period, freq="h") # Почасовая разметка на будующий месяц
    forecast = m.predict(future) # Предсказание будущих значений CLOSE (dataframe в котором приситсвуют все данные df_train + предсказанные данные)

    for row in forecast.itertuples():
        if row[0] > len(df_train.index): # Первые X строчек всегда одиннаковые - смтр. стр. 38
            # Сравнение данных - если подходят по времени (после 12 мая, до 11 июня, исключая дни, когда закрыта биржа, после 10 часов утра и до полуночи) то они добавляются в массив data_final
            time_string = row[1].strftime('%Y-%m-%d %X')
            time_string = time_string.split(" ")
            time_string_copy = time_string[1].split(":")
            time_string = time_string[0].split("-")
            year = int(time_string[0])
            month = int(time_string[1])
            day = int(time_string[2])
            hour = int(time_string_copy[0])
            if hour > 9 and hour != 19: # Сравнение
                if month == 5 and day > 12 and day not in non_assessable_days or month == 6 and day < 11 and day not in non_assessable_days or day == 10 and month == 6: # Сравнение
                    if day < 10: # Добавляет 0 перед днем для правильного формата
                        data_final.append([len(data_final) - 1, ticker, f"{year}-0{month}-0{day}", f"{hour}:00:00", row[2]]) # Добавление
                    else:
                        data_final.append([len(data_final) - 1, ticker, f"{year}-0{month}-{day}", f"{hour}:00:00", row[2]]) # Добавление


# Переносит все данные в массив stock_prediction.csv
with open("stock_prediction.csv", "w", newline="") as output: #Неформатированные данные
    writer = csv.writer(output, delimiter=",")
    writer.writerows(data_final)


def scale_format_data(ticker, data): # Приближает данные к последней цене - поскольку цена отходит от истинной цены пока рынки не действуют - смтр. стр. 9
    return int((company_final_known_stock_dict[ticker] * 4 + data) / 5 * 100) / 100


# Данные последней цены - 10/05/2024 23:49
company_final_known_stock_dict = {"CHMF": 1960.0,
                                  "VKCO": 574.8,
                                  "MTSS": 312.15,
                                  "SBER": 313.49,
                                  "SMLT": 3628.0,
                                  "YNDX": 4356.2,
                                  "AGRO": 1577.0,
                                  "SIBN": 751.0}

# Перевод данных в формат, востребованным в файле sample_submission.csv
unformatted_data = pd.read_csv("stock_prediction.csv", delimiter=",")

# Формат №1 - С параметром KEY
data_format = pd.read_csv("sample_submission.csv", delimiter=",")

submission_data_final = [["KEY", "TICKER", "DATE", "TIME", "CLOSE"]]

for row in data_format.itertuples():
    for unformatted_row in unformatted_data.itertuples():
        # Сравниваются данные под индексами TICKER, DATE и TIME - если отличий нет, то файлы копируются в массив submission_data_finаl
        if row[2] == unformatted_row[2] and row[3] == unformatted_row[3] and row[4] == unformatted_row[4]: # Проверка
            submission_data_final.append([row[1], row[2], row[3], row[4], scale_format_data(row[2], unformatted_row[5])]) # Добавление + Форматирование цены
            break; # Если данные находятся то заранее происходит выход из цикла

# Переносит все данные в массив stock_prediction_submission.csv
with open("stock_prediction_submission.csv", "w", newline="") as output: #Форматированные данные
    writer = csv.writer(output, delimiter=",")
    writer.writerows(submission_data_final)

# Формат №2 - Без параметра KEY
# data_format = pd.read_csv("test2.csv", delimiter=";")
#
# submission_data_final = [["TICKER", "DATE", "TIME", "CLOSE"]]
#
# for row in data_format.itertuples():
#     for unformatted_row in unformatted_data.itertuples():
#         if row[1] == unformatted_row[2] and row[2] == unformatted_row[3] and row[3] == unformatted_row[4]:
#             submission_data_final.append([row[1], row[2], row[3], scale_format_data(row[1], unformatted_row[5])])
#             break;
#
# with open("test2.csv", "w", newline="") as output:
#     writer = csv.writer(output, delimiter=",")
#     writer.writerows(submission_data_final)
