import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib
from collections import namedtuple
import matplotlib as mpl

mpl.style.use("figPublish")
# mpl.rcParams['axes.linewidth'] = 2

cmap = matplotlib.cm.get_cmap("hot_r")


from callfunc import processData

basePath = [
    "/data/Clustering/SleepDeprivation/RatJ/Day1/",
    "/data/Clustering/SleepDeprivation/RatK/Day1/",
    "/data/Clustering/SleepDeprivation/RatN/Day1/",
    "/data/Clustering/SleepDeprivation/RatJ/Day2/",
    "/data/Clustering/SleepDeprivation/RatK/Day2/",
    "/data/Clustering/SleepDeprivation/RatN/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day4/"
]


sessions = [processData(_) for _ in basePath]

tbin = lambda x: np.linspace(x - 5, x + 5, 40)
plt.clf()
for sub, sess in enumerate(sessions):

    sess.trange = np.array([])
    peakrpl = sess.ripple.peaktime
    peakspndl = sess.spindle.peaktime
    # peakspndl = sess.swa.time

    hist_rpl = [np.histogram(peakrpl, bins=tbin(_))[0] for _ in peakspndl]

    t = tbin(0)[:-1]
    hist_rpl = np.asarray(hist_rpl).sum(axis=0)
    plt.subplot(2, 3, sub + 1)

    plt.plot(t, hist_rpl, "#2d3143")

    subname = sess.sessinfo.session.name
    day = sess.sessinfo.session.day
    if sub < 3:
        plt.title(f"{subname} {day} SD")
    else:
        plt.title(f"{subname} {day} NSD")


plt.subplot(2, 3, 4)
plt.ylabel("Counts", fontweight="bold")
plt.xlabel("Time from spindle peak onset (sec)", fontweight="bold")

plt.suptitle("Ripples around Spindles ")
