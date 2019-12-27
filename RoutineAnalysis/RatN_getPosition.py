import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


fileName = "/data/Clustering/SleepDeprivation/RatN/Day2/position/Take 2019-10-11 07.19.00 AM.csv"
check_file = pd.read_csv(fileName, header=5)


class ExtractPsoiton:

    nChans = 134
    sRate = 30000
    binSize = 0.250  # in seconds
    timeWindow = 3600  # in seconds (15 minutes)

    def __init__(self, basePath):
        # self.sessionnName = os.path.basename(os.path.normpath(basePath))
        self.sessionName = basePath.split("/")[-3] + basePath.split("/")[-2]
        self.basePath = basePath

    def plotPosition(self):

        positionStruct = pd.read_csv(self.basePath, header=5)
        positionStruct = positionStruct.iloc[:, [1, 6, 7, 8]]
        positionStruct.interpolate(axis=0)

        posX = positionStruct.iloc[:, 1]
        posY = positionStruct.iloc[:, 2]
        posZ = positionStruct.iloc[:, 3]

        # plt.clf()
        # plt.plot(posX, posZ, ".")

    def Velocity(self):
        


RatNDay2 = ExtractPsoiton(fileName).plotPosition()

