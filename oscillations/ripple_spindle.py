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

tbin = lambda x: np.linspace(x - 20, x + 20, 40)
plt.clf()
for sub, sess in enumerate(sessions):

    sess.trange = np.array([])
    peakrpl = sess.ripple.peaktime
    peakspndl = sess.spindle.peaktime

    hist_rpl = [np.histogram(peakrpl, bins=tbin(_))[0] for _ in peakspndl]


t = tbin(0)[:-1] + 0.5
hist_rpl = np.asarray(hist_rpl).sum(axis=0)
plt.plot(t, hist_rpl)
