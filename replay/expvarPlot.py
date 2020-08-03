#%% Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter

from callfunc import processData

basePath = [
    # "/data/Clustering/SleepDeprivation/RatJ/Day1/",
    # "/data/Clustering/SleepDeprivation/RatK/Day1/",
    # "/data/Clustering/SleepDeprivation/RatN/Day1/",
    # "/data/Clustering/SleepDeprivation/RatJ/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day2/",
    # "/data/Clustering/SleepDeprivation/RatN/Day2/",
    "/data/Clustering/SleepDeprivation/RatK/Day4/",
]


sessions = [processData(_) for _ in basePath]

#%% Explained variance PRE-MAZE-POST
# region
"""Only found stable units for 3 sessions
"""
plt.clf()
fig = plt.figure(1, figsize=(6, 10))
gs = gridspec.GridSpec(3, 1, figure=fig)
fig.subplots_adjust(hspace=0.4)

for sub, sess in enumerate(sessions):

    sess.trange = np.array([])
    # sess.spikes.extract()
    sess.spikes.stability.firingRate()
    # sess.spikes.stability.refPeriodViolation()
    ev1, rev1 = sess.replay.expvar()
    # sess.brainstates.detect()
    # violations = sess.spikes.stability.violations
    axstate = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[sub], hspace=0.2)

    ax1 = fig.add_subplot(axstate[1:])
    t = (np.linspace(0, 40, 41) * 0.25)[1:] - 0.125
    # sessions[0].brainstates.addBackgroundtoPlots(ax1)
    ax1.fill_between(
        t,
        np.mean(ev1.squeeze(), axis=0) - np.std(ev1.squeeze(), axis=0),
        np.mean(ev1.squeeze(), axis=0) + np.std(ev1.squeeze(), axis=0),
        color="#7c7979",
    )
    ax1.fill_between(
        t,
        np.mean(rev1.squeeze(), axis=0) - np.std(rev1.squeeze(), axis=0),
        np.mean(rev1.squeeze(), axis=0) + np.std(rev1.squeeze(), axis=0),
        color="#87d498",
    )
    ax1.plot(t, np.mean(ev1.squeeze(), axis=0), "k")
    ax1.plot(t, np.mean(rev1.squeeze(), axis=0), "#02c59b")
    ax1.set_xlabel("Time (h)")
    ax1.set_ylabel("Explained variance")
    ax1.legend(["EV", "REV"])
    ax1.text(0.2, 0.28, "POST SD", fontweight="bold")
    ax1.set_xlim([0, 10])

    axhypno = fig.add_subplot(axstate[0], sharex=ax1)
    sess.viewdata.hypnogram(ax1=axhypno)
    # panel_label(axhypno, "a")
    # ax1.set_ylim([0, 0.3])


# endregion
