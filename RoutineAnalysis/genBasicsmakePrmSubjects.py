import numpy as np
import os
from MakePrmKlusta import makePrmPrb
from makeChanMap import ExtractChanXml
from pathlib import Path

basePath = [
    # "/data/Clustering/SleepDeprivation/RatJ/Day1/",
    # "/data/Clustering/SleepDeprivation/RatJ/Day2/",
    "/data/Clustering/SleepDeprivation/RatJ/Day3/",
    # "/data/Clustering/SleepDeprivation/RatK/Day1/",
    # "/data/Clustering/SleepDeprivation/RatK/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day3/",
    # "/data/Clustering/SleepDeprivation/RatK/Day4/",
    # "/data/Clustering/SleepDeprivation/RatN/Day1/",
    # "/data/Clustering/SleepDeprivation/RatN/Day2/",
    # "/data/Clustering/SleepDeprivation/RatN/Day3/",
    # "/data/Clustering/SleepDeprivation/RatN/Day4/",
]
nSessions = len(basePath)

basics = [ExtractChanXml(basePath[i]) for i in range(nSessions)]

prmGen = [makePrmPrb(basePath[i]) for i in range(nSessions)]

for sess in prmGen:
    # prmGen[i].makePrm()
    # prmGen[i].makePrb()
    # prmGen[i].makePrmServer()
    # prmGen[i].makePrbServer()
    sess.makePrbCircus(probetype="buzsaki")

