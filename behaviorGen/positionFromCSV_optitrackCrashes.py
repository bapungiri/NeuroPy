import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

framef1 = "/data/Clustering/SleepDeprivation/test/frameInfo 2020-12-02 10.54.39 AM.csv"
posf1 = "/data/Clustering/SleepDeprivation/test/posInfo 2020-12-02 10.54.39 AM.csv"

framef2 = "/data/Clustering/SleepDeprivation/test/frameInfo 2020-12-02 11.17.43 AM.csv"
posf2 = "/data/Clustering/SleepDeprivation/test/posInfo 2020-12-02 11.17.43 AM.csv"

frame1 = pd.read_csv(framef1, header=0)
pos1 = pd.read_csv(posf1, header=0)

frame2 = pd.read_csv(framef2, header=0)
pos2 = pd.read_csv(posf2, header=0)

framedata = pd.concat([frame1, frame2]).reset_index()
posdata = pd.concat([pos1, pos2]).reset_index()
