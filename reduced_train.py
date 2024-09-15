import csv
import pandas as pd

data = [["KEY", "TICKER", "DATE", "TIME", "OPEN", "HIGH", "LOW", "CLOSE", "VOL"]]

data_points = pd.read_csv("train.csv", delimiter=";")
data_points = data_points.reset_index()

for tuple in data_points.itertuples():
    time_list = tuple[5].split(":")
    if int(time_list[1]) == 0:
        data.append([tuple[0], tuple[2], tuple[4], tuple[5], tuple[6], tuple[7], tuple[8], tuple[9], tuple[10]])

with open("reduced_train.csv", "w", newline="") as output:
    writer = csv.writer(output, delimiter=",")
    writer.writerows(data)