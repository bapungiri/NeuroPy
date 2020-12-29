import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
import csv

# file = "/data/Clustering/SleepDeprivation/RatS/Day3SD/position/Take 2020-11-29 11.03.45 AM.csv"

# data = pd.read_csv(file, skiprows=6, skip_blank_lines=False, usecols=[1, 6, 7, 8])

# with open(file, newline="") as a:

#     reader = csv.reader(a, delimiter=",")
#     line_count = 0
#     for row in reader:
#         # print(row)
#         if "Rigid Body" in row:
#             loc1 = [_ for _ in range(len(row)) if row[_] == "Rigid Body"]

#         if "Position" in row:
#             loc2 = [_ for _ in range(len(row)) if row[_] == "Position"]
#             break
#         line_count += 1

# file1 = "/run/media/bapung/mercik/RatS/Day3SD/2020-11-29_07-53-30/experiment1/recording1/continuous/Intan_Rec._Controller-100.0/continuous.dat"

# file2 = "/run/media/bapung/mercik/RatS/Day3SD/2020-11-29_11-03-40/experiment1/recording1/continuous/Intan_Rec._Controller-100.0/continuous.dat"

# file3 = "/run/media/bapung/mercik/RatS/Day3SD/2020-11-29_12-22-17/experiment1/recording1/continuous/Intan_Rec._Controller-100.0/continuous.dat"

# file4 = "/run/media/bapung/mercik/RatS/Day3SD/2020-11-29_16-01-30/experiment1/recording1/continuous/Intan_Rec._Controller-100.0/continuous.dat"

# file5 = "/run/media/bapung/mercik/RatS/Day3SD/2020-11-29_21-49-14/experiment1/recording1/continuous/Intan_Rec._Controller-100.0/continuous.dat"


# a = []
# for file in [file1, file2, file3, file4, file5]:
#     val = len(np.memmap(file, dtype="int16", mode="r")) / (30000 * 195)
#     a.append(val)


# a = np.asarray(a)


file = "/data/Clustering/SleepDeprivation/RatN/Day2/RatN_Day2_2019-10-11_03-58-54_position.npy"

data = np.load(file, allow_pickle=True).item()
# data = pd.read_csv(file, header=5)
