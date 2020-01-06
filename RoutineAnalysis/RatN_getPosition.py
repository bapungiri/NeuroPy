import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

fileName = "/data/Clustering/SleepDeprivation/RatN/Day2/position/RatNDay2-2019-10-11 07.19.00 AM.csv"
check_file = pd.read_csv(fileName, header=5)
firstLine = pd.read_csv(fileName, nrows=0).iloc[:, 3]

with open(fileName, newline="") as f:
    reader = csv.reader(f)
    row1 = next(reader)
    StartTime = [
        row1[i + 1] for i in range(len(row1)) if row1[i] == "Capture Start Time"
    ]


class ExtractPosition:

    nChans = 134
    sRate = 30000
    binSize = 0.250  # in seconds
    timeWindow = 3600  # Number of bins (15 minutes)

    def __init__(self, basePath):
        # self.sessionName = os.path.basename(os.path.normpath(basePath))
        self.sessionName = basePath.split("/")[-3] + basePath.split("/")[-2]
        self.basePath = basePath
        positionStruct = pd.read_csv(basePath, header=5)
        # TODO get automatic column location
        positionStruct = positionStruct.iloc[:, [1, 6, 7, 8]]
        positionStruct.interpolate(axis=0)

        self.time = positionStruct.iloc[:, 0]
        self.posX = positionStruct.iloc[:, 1]
        self.posY = positionStruct.iloc[:, 2]
        self.posZ = positionStruct.iloc[:, 3]
        self.dt = self.time[1] - self.time[0]

    def plotPosition(self):

        plt.clf()
        plt.plot(self.posX, self.posZ, ".")

    def Speed(self):
        location = np.sqrt((self.posZ) ** 2 + (self.posX) ** 2)
        spd = np.abs(np.diff(location)) / self.dt

        self.speed = spd.tolist()
        return self.speed


RatNDay2 = ExtractPosition(fileName)
velocity = RatNDay2.Speed()
plt.clf()
plt.plot(velocity)
