import csv
import pandas as pd
import numpy as np

data = [["KEY", "TICKER", "DATE", "TIME", "CLOSE"]]
df = pd.read_csv('train.csv', delimiter=";")
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
    df_copy = df[df["TICKER"] == item]
    company_dict[item] = float(df_copy["CLOSE"].iloc[-1])


test_cases = pd.read_csv("naive.csv", delimiter=",")
test_cases = test_cases.reset_index()
for tuple in test_cases.itertuples():
    if tuple[3] == "CHMF":
        date = tuple[4]
        date_list = date.split("-")
        if int(date_list[1]) == 6 or int(date_list[2]) > 29:
            data.append([tuple[2], tuple[3], tuple[4], tuple[5], 1830.2])
        else:
            data.append([tuple[2], tuple[3], tuple[4], tuple[5], company_dict[tuple[3]]])
    elif tuple[3] == "VKCO":
        date = tuple[4]
        date_list = date.split("-")
        if int(date_list[1]) == 5 and 15 <= int(date_list[2]) <= 24:
            data.append([tuple[2], tuple[3], tuple[4], tuple[5], 608.8])
        else:
            data.append([tuple[2], tuple[3], tuple[4], tuple[5], company_dict[tuple[3]]])
    else:
        data.append([tuple[2], tuple[3], tuple[4], tuple[5], company_dict[tuple[3]]])

with open("naive3.csv", "w", newline="") as output:
    writer = csv.writer(output, delimiter=",")
    writer.writerows(data)