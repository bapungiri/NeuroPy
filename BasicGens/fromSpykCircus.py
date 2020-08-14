import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter

from callfunc import processData

basePath = [
    # "/data/Clustering/SleepDeprivation/RatJ/Day1/",
    # "/data/Clustering/SleepDeprivation/RatK/Day1/",
    # "/data/Clustering/SleepDeprivation/RatN/Day1/",
    # "/data/Clustering/SleepDeprivation/RatJ/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day2/",
    # "/data/Clustering/SleepDeprivation/RatN/Day2/"
    # "/data/Clustering/SleepDeprivation/RatJ/Day4/"
    # "/data/Clustering/SleepDeprivation/RatK/Day4/"
    "/data/Clustering/SleepDeprivation/RatN/Day4/"
]


sessions = [processData(_) for _ in basePath]

for sub, sess in enumerate(sessions):

    sess.trange = np.array([])

    sess.spikes.fromCircus(fileformat="same_folder")
    # print(len(sess.spikes.times))
    # sess.placefield.pf2d.compute()
    # sess.placefield.pf2d.plot()

#     sess.spikes.stability.firingRate()


# stability = sess.spikes.stability.isStable
# unstable = sess.spikes.stability.unstable
# stable = sess.spikes.stability.stable
