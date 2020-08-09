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
    "/data/Clustering/SleepDeprivation/RatN/Day1/",
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
    sess.replay.expvar.compute()
    # sess.brainstates.detect()
    # violations = sess.spikes.stability.violations
    axstate = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[sub], hspace=0.2)

    ax1 = fig.add_subplot(axstate[1:])
    sess.replay.expvar.plot(ax=ax1)

    axhypno = fig.add_subplot(axstate[0], sharex=ax1)
    sess.brainstates.hypnogram(ax1=axhypno, tstart=sess.epochs.post[0], unit="h")
    # panel_label(axhypno, "a")
    # ax1.set_ylim([0, 0.3])


# endregion
