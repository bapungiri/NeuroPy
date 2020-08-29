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
    "/data/Clustering/SleepDeprivation/RatK/Day1/",
    "/data/Clustering/SleepDeprivation/RatN/Day1/",
    # "/data/Clustering/SleepDeprivation/RatJ/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day2/",
    # "/data/Clustering/SleepDeprivation/RatN/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day4/",
    # "/data/Clustering/SleepDeprivation/RatN/Day4/",
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
    maze = sess.epochs.maze
    post = sess.epochs.post
    bins = [
        [maze[0], maze[1]],
        [post[0] + 0 * 3600, post[0] + 1 * 3600],
        [post[0] + 5 * 3600, post[0] + 8 * 3600],
    ]
    sess.spikes.stability.firingRate(bins=bins)
    # sess.spikes.stability.refPeriodViolation()
    # violations = sess.spikes.stability.violations
    sess.replay.expvar.compute(template=bins[1], match=bins[2], control=bins[0])

    axstate = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[sub], hspace=0.2)

    ax1 = fig.add_subplot(axstate[1:])
    sess.replay.expvar.plot(ax=ax1, tstart=bins[2][0])

    axhypno = fig.add_subplot(axstate[0], sharex=ax1)
    sess.brainstates.hypnogram(ax1=axhypno, tstart=bins[2][0], unit="h")
    # panel_label(axhypno, "a")
    # ax1.set_ylim([0, 0.3])


# endregion

#%% Explained variance during recovery sleep while controlling for MAZE correlations
# region
"""Only found stable units for 3 sessions
"""
plt.clf()
fig = plt.figure(1, figsize=(6, 10))
gs = gridspec.GridSpec(3, 1, figure=fig)
fig.subplots_adjust(hspace=0.4)

for sub, sess in enumerate(sessions):

    sess.trange = np.array([])
    maze = sess.epochs.maze
    post = sess.epochs.post
    bins = [
        [post[0], post[0] + 3600],
        [post[0] + 4 * 3600, post[0] + 5 * 3600],
        [post[0] + 5 * 3600, post[0] + 6 * 3600],
    ]
    sess.spikes.stability.firingRate(bins=bins)
    # sess.spikes.stability.refPeriodViolation()
    # violations = sess.spikes.stability.violations
    sess.replay.expvar.compute(template=bins[1], match=bins[2], control=bins[0])

    axstate = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[sub], hspace=0.2)

    ax1 = fig.add_subplot(axstate[1:])
    sess.replay.expvar.plot(ax=ax1, tstart=post[0])

    axhypno = fig.add_subplot(axstate[0], sharex=ax1)
    sess.brainstates.hypnogram(ax1=axhypno, tstart=post[0], unit="h")
    axhypno.set_title(sess.sessinfo.session.sessionName)
    # panel_label(axhypno, "a")
    # ax1.set_ylim([0, 0.3])


# endregion
