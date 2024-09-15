import csv #Позволяет взаимодействовать с файлами формата csv (писать данные в файлы)
import pandas as pd # Pandas — главная библиотека в Python для работы с данными. (Позволяет взаимодействовать с файлами формата csv (читать данные в файлах))
from prophet import Prophet #Prophet — библиотека, позволяющая диагностировать

stocks = ('VKCO', 'SBER', 'YNDX', 'CHMF', 'MTSS', 'SMLT', 'AGRO', 'SIBN')

period = 31 * 24


def load_data(ticker):
    data = train_data[train_data["TICKER"] == ticker]
    data.reset_index(inplace=True)
    return data


train_data = pd.read_csv("train.csv", delimiter=";")
train_data['datetime'] = pd.to_datetime(train_data['DATE']+' '+train_data['TIME'])

data_final = [["KEY", "TICKER", "DATE", "TIME", "CLOSE"]]
non_assessable_days = [10, 11 ,12 ,18 ,19, 25, 26, 1, 2, 8, 9]

for ticker in stocks:
    data = load_data(ticker)

    # Predict forecast with Prophet.
    df_train = data[['datetime', 'CLOSE']]
    print(df_train)
    df_train = df_train.rename(columns={"datetime": "ds", "CLOSE": "y"})

    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period, freq="h")
    forecast = m.predict(future)

    for row in forecast.itertuples():
        if row[0] > 50000:
            time_string = row[1].strftime('%Y-%m-%d %X')
            time_string = time_string.split(" ")
            time_string_copy = time_string[1].split(":")
            time_string = time_string[0].split("-")
            year = int(time_string[0])
            month = int(time_string[1])
            day = int(time_string[2])
            hour = int(time_string_copy[0])
            if hour > 9 and hour != 19:
                if month == 5 and day > 12 and day not in non_assessable_days or month == 6 and day < 11 and day not in non_assessable_days or day == 10 and month == 6:
                    if day < 10:
                        data_final.append([len(data_final) - 1, ticker, f"{year}-0{month}-0{day}", f"{hour}:00:00", row[2]])
                    else:
                        data_final.append([len(data_final) - 1, ticker, f"{year}-0{month}-{day}", f"{hour}:00:00", row[2]])


with open("stock_prediction.csv", "w", newline="") as output:
    writer = csv.writer(output, delimiter=",")
    writer.writerows(data_final)


def scale_format_data(ticker, data):
    return int((company_final_known_stock_dict[ticker] * 4 + data) / 5 * 100) / 100


company_final_known_stock_dict = {"CHMF": 1960.0,
                                  "VKCO": 574.8,
                                  "MTSS": 312.15,
                                  "SBER": 313.49,
                                  "SMLT": 3628.0,
                                  "YNDX": 4356.2,
                                  "AGRO":1577.0,
                                  "SIBN":751.0}

unformatted_data = pd.read_csv("stock_prediction.csv", delimiter=",")
data_format = pd.read_csv("sample_submission.csv", delimiter=",")

submission_data_final = [["KEY", "TICKER", "DATE", "TIME", "CLOSE"]]

for row in data_format.itertuples():
    for unformatted_row in unformatted_data.itertuples():
        if row[2] == unformatted_row[2] and row[3] == unformatted_row[3] and row[4] == unformatted_row[4]:
            submission_data_final.append([row[1], row[2], row[3], row[4], scale_format_data(row[2], unformatted_row[5])])
            break;

with open("stock_prediction_submission.csv", "w", newline="") as output:
    writer = csv.writer(output, delimiter=",")
    writer.writerows(submission_data_final)

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
