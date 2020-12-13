import numpy as np
import time
import matplotlib.pyplot as plt

file = "/data/Clustering/SleepDeprivation/RatS/Day3SD/RatS_Day3SD_2020-11-29_07-53-30.probe.npy"

data = np.load(file, allow_pickle=True).item()

# for (key, val) in data.items():
#     print(key, val)
