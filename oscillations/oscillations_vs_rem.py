import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib
from collections import namedtuple
from scipy.ndimage import gaussian_filter1d

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
plt.clf()
tbin = lambda x: np.arange(x - 80, x + 80)
for sub, sess in enumerate(sessions):

    sess.trange = np.array([])
    peakrpl = sess.ripple.peaktime
    peakspndl = sess.spindle.peaktime
    # hswa = sess.swa.time
    states = sess.brainstates.states
    remstart = states[states["name"] == "rem"].start.values

    rpl_rem = [np.histogram(peakrpl, bins=tbin(_))[0] for _ in remstart]
    spndl_rem = [np.histogram(peakspndl, bins=tbin(_))[0] for _ in remstart]

    t = tbin(0)[:-1] + 0.5
    rpl_rem = np.asarray(rpl_rem).sum(axis=0)
    spndl_rem = np.asarray(spndl_rem).sum(axis=0)

    plt.subplot(2, 3, sub + 1)
    plt.plot(t, spndl_rem, "blue", linewidth=2)
    plt.plot(t, rpl_rem, "r")

    subname = sess.sessinfo.session.name
    day = sess.sessinfo.session.day
    if sub < 3:
        plt.title(f"{subname} {day} SD")
    else:
        plt.title(f"{subname} {day} NSD")


plt.subplot(2, 3, 4)
plt.ylabel("Counts")
plt.xlabel("Time from REM onset (sec)")
plt.legend(["spindles", "ripples"])

plt.suptitle("Ripple and Spindles aroung REM sleep")
