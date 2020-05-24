import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter
from getPosition import posfromFBX
from mathutil import getICA_Assembly
from callfunc import processData

basePath = [
    # "/data/Clustering/SleepDeprivation/RatJ/Day1/",
    # "/data/Clustering/SleepDeprivation/RatK/Day1/",
    # "/data/Clustering/SleepDeprivation/RatN/Day1/",
    # "/data/Clustering/SleepDeprivation/RatN/Day1/",
    # "/data/Clustering/SleepDeprivation/RatJ/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day2/",
    # "/data/Clustering/SleepDeprivation/RatN/Day2/",
    # "/data/Clustering/SleepDeprivation/RatJ/Day3/",
    "/data/Clustering/SleepDeprivation/RatK/Day3/",
    # "/data/Clustering/SleepDeprivation/RatK/Day4/"
]


sessions = [processData(_) for _ in basePath]

# a = np.arange(16).reshape(4, 4)
# V, W = getICA_Assembly(a)

for sub, sess in enumerate(sessions):
    # sess.makePrmPrb.makePrbCircus("diagbio")
    sess.trange = np.array([])
    # sess.spikes.fromCircus(fileformat="diff_folder")
    # sess.placefield.pf2d.compute()
    # # # sess.placefield.pf2d.plotRaw()
    # sess.placefield.pf2d.plotMap()

    sess.position.getPosition()


# file = "/data/Clustering/SleepDeprivation/RatK/Day3/RatK_Day3_2019-08-10_04-33-10_timestamps.npy"

# fg = pd.read_csv(file, index_col=0)
# fg = np.load(file)
# fg = np.memmap(file, dtype="int16", mode="r")
