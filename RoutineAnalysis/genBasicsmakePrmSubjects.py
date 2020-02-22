import numpy as np
import os
from MakePrmKlusta import makePrmPrb
from makeChanMap import ExtractChanXml
from pathlib import Path

basePath = [
    # "/data/Clustering/SleepDeprivation/RatJ/Day1/",
    # "/data/Clustering/SleepDeprivation/RatJ/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day1/",
    # "/data/Clustering/SleepDeprivation/RatK/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day3/",
    # "/data/Clustering/SleepDeprivation/RatK/Day4/",
    "/data/Clustering/SleepDeprivation/RatN/Day1/",
    # "/data/Clustering/SleepDeprivation/RatN/Day2/",
    # "/data/Clustering/SleepDeprivation/RatN/Day3/",
    # "/data/Clustering/SleepDeprivation/RatN/Day4/",
]
nSessions = len(basePath)
shanks_all = [8]
shanks_Chan_all = [16]

# basics = [ExtractChanXml(basePath[i]) for i in range(nSessions)]
prmGen = [makePrmPrb(basePath[i]) for i in range(nSessions)]

# for i in range(nSessions):
# prmGen[i].makePrm()
# prmGen[i].makePrb()
#     # prmGen[i].makePrmServer()
#     # prmGen[i].makePrbServer()
