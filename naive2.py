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
    data.append([tuple[2], tuple[3], tuple[4], tuple[5], company_dict[tuple[3]]])
with open("naive2.csv", "w", newline="") as output:
    writer = csv.writer(output, delimiter=",")
    writer.writerows(data)