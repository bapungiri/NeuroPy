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
    "/data/Clustering/SleepDeprivation/RatJ/Day1/",
    "/data/Clustering/SleepDeprivation/RatK/Day1/",
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
    duration = np.diff(ripples, axis=1).squeeze()
    peakpower = sess.ripple.peakpower
    tstart = sess.epochs.post[0]
    tend = sess.epochs.post[0] + 5 * 3600
    nbin = 5
    tbin = np.linspace(tstart, tend, nbin + 1)
    colors = [cmap(_) for _ in np.linspace(0, 1, nbin)]
    nripples = []
    ripple_dur = []
    ripple_cat = []
    peakpower_all = []
    for i in range(nbin):
        binstart = tbin[i]
        binend = tbin[i + 1]

        ripple_ind = np.where((ripples[:, 0] > binstart) & (ripples[:, 0] < binend))[0]
        dur_bin = duration[ripple_ind]
        bin_cat = (i + 1) * np.ones(len(dur_bin))
        power_bin = peakpower[ripple_ind]
        # peakpowerbin = peakpower[ripple_ind]
        # powerbinning = np.logspace(np.log10(1.2), np.log10(60), 50)
        # powerbinning = np.linspace(5, 40,)
        # peakhist, _ = np.histogram(peakpowerbin, bins=powerbinning)
        # peakhist = peakhist / np.sum(peakhist)
        # plt.plot(powerbinning[:-1], peakhist, color=colors[i])
        # plt.yscale("log")
        nripples.append(len(ripple_ind))
        ripple_dur.extend(dur_bin)
        ripple_cat.extend(bin_cat)
        peakpower_all.extend(power_bin)

    nripples = np.array(nripples)
    data = pd.DataFrame(
        {"dur": ripple_dur, "hour": ripple_cat, "peakpower": peakpower_all}
    )
    subname = sess.sessinfo.session.name
    day = sess.sessinfo.session.day

    plt.subplot(3, 3, sub + 1)
    # plt.bar(np.arange(0.5, 5.5, 1), nripples)
    sns.countplot(x="hour", data=data, color="#f0b67f")
    plt.title(f"{subname} {day} SD")

    plt.subplot(3, 3, sub + 3 + 1)
    sns.violinplot(x="hour", y="dur", data=data, color="#4CAF50")
    # plt.plot(ripple_dur)
    plt.ylabel("duration")

    plt.subplot(3, 3, sub + 6 + 1)
    sns.violinplot(x="hour", y="peakpower", data=data, color="#CE93D8")
    # plt.plot(ripple_dur)
    plt.ylabel("duration")

plt.suptitle("Ripples during Sleep deprivation period")
