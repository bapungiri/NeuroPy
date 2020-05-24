import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib
from collections import namedtuple

cmap = matplotlib.cm.get_cmap("hot_r")


from callfunc import processData

basePath = [
    # "/data/Clustering/SleepDeprivation/RatJ/Day1/",
    # "/data/Clustering/SleepDeprivation/RatK/Day1/",
    "/data/Clustering/SleepDeprivation/RatN/Day1/",
    # "/data/Clustering/SleepDeprivation/RatJ/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day2/",
    # "/data/Clustering/SleepDeprivation/RatN/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day4/"
]


sessions = [processData(_) for _ in basePath]


plt.clf()
group = []
for sub, sess in enumerate(sessions):

    sess.trange = np.array([])

    ripples = sess.ripple.time
    peakpower = sess.ripple.peakpower
    tstart = sess.epochs.post[0]
    tend = sess.epochs.post[0] + 5 * 3600
    nbin = 10
    tbin = np.linspace(tstart, tend, nbin + 1)
    colors = [cmap(_) for _ in np.linspace(0, 1, nbin)]
    for i in range(nbin):
        binstart = tbin[i]
        binend = tbin[i + 1]

        ripple_ind = np.where((ripples[:, 0] > binstart) & (ripples[:, 0] < binend))[0]
        peakpowerbin = peakpower[ripple_ind]
        # powerbinning = np.logspace(np.log10(1.2), np.log10(60), 50)
        powerbinning = np.linspace(5, 40, 31)
        peakhist, _ = np.histogram(peakpowerbin, bins=powerbinning)
        plt.plot(powerbinning[:-1], peakhist, color=colors[i])
        # plt.yscale("log")
