#%% Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter
import warnings
from callfunc import processData

warnings.simplefilter(action="default")

basePath = [
    # "/data/Clustering/SleepDeprivation/RatJ/Day1/",
    "/data/Clustering/SleepDeprivation/RatK/Day1/",
    "/data/Clustering/SleepDeprivation/RatN/Day1/",
    # "/data/Clustering/SleepDeprivation/RatJ/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day2/",
    "/data/Clustering/SleepDeprivation/RatN/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day4/",
    # "/data/Clustering/SleepDeprivation/RatN/Day4/",
]


sessions = [processData(_) for _ in basePath]

plt.clf()
fig = plt.figure(1, figsize=(8.5, 11))
gs = gridspec.GridSpec(5, 5, figure=fig)
fig.subplots_adjust(hspace=0.3, wspace=0.5)
fig.suptitle("Sleep states related analysis")
titlesize = 8
panel_label = lambda ax, label: ax.text(
    x=-0.08,
    y=1.15,
    s=label,
    transform=ax.transAxes,
    fontsize=12,
    fontweight="bold",
    va="top",
    ha="right",
)

#%% Average explained variance during sleep deprivation

# region
"""Only found stable units for 3 sessions
"""

for sub, sess in enumerate(sessions):

    sess.trange = np.array([])
    maze = sess.epochs.maze
    post = sess.epochs.post
    pre = sess.epochs.pre

    bins = [
        [pre[1] - 1 * 3600, pre[1]],
        [maze[0], maze[1]],
        [post[0], post[0] + 1 * 3600],
        [post[0] + 1 * 3600, post[0] + 2 * 3600],
        [post[0] + 2 * 3600, post[0] + 3 * 3600],
        [post[0] + 3 * 3600, post[0] + 4 * 3600],
        [post[0] + 4 * 3600, post[0] + 5 * 3600],
    ]
    sess.spikes.stability.firingRate(bins=bins)
    sess.replay.expvar.compute(
        template=bins[1], match=[post[0], post[0] + 5 * 3600], control=bins[0],
    )

    # axstate = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[sub], hspace=0.2)

    ax1 = plt.subplot(gs[0])
    sess.replay.expvar.plot(ax=ax1, tstart=post[0])
    # if sub == 0:
    #     ax1.set_ylim([0, 0.55])
    # else:
    #     ax1.set_ylim([0, 0.4])

    # if i > 0:
    #     ax1.spines["left"].set_visible(False)
    #     ax1.set_yticks([])
    #     ax1.set_yticklabels([])
    #     ax1.set_ylabel("")
    #     ax1.legend("")
    #     ax1.set_title("")

    axhypno = fig.add_subplot(axstate[0], sharex=ax1)
    sess.brainstates.hypnogram(ax1=axhypno, tstart=post[0], unit="h")
    axhypno.set_title(sess.sessinfo.session.sessionName)
    # panel_label(axhypno, "a")
    # ax1.set_ylim([0, 0.3])


# endregion


#%% Explained variance during recovery sleep while controlling for MAZE correlations

# region
"""Only found stable units for 3 sessions
"""

for sub, sess in enumerate(sessions):

    sess.trange = np.array([])
    maze = sess.epochs.maze
    post = sess.epochs.post
    pre = sess.epochs.pre

    # for i, hour in enumerate(range(5, 10)):
    bins = [
        # [pre[1] - 1 * 3600, pre[1]],
        # [maze[0], maze[1]],
        [post[0], post[0] + 1 * 3600],
        # [post[0] + 1 * 3600, post[0] + 2 * 3600],
        # [post[0] + 2 * 3600, post[0] + 3 * 3600],
        # [post[0] + 3 * 3600, post[0] + 4 * 3600],
        [post[0] + 4 * 3600, post[0] + 5 * 3600],
        [post[0] + 5 * 3600, post[0] + 6 * 3600],
        # [post[0] + 6 * 3600, post[0] + 7 * 3600],
        # [post[0] + 7 * 3600, post[0] + 8 * 3600],
        # [post[0] + 8 * 3600, post[0] + 9 * 3600],
        # [post[0] + 9 * 3600, post[0] + 10 * 3600],
    ]
    # sess.spikes.stability.refPeriodViolation()
    # violations = sess.spikes.stability.violations
    sess.spikes.stability.firingRate(bins=bins)
    sess.replay.expvar.compute(
        template=bins[1],
        match=[post[0] + 5 * 3600, post[0] + 6 * 3600],
        control=bins[0],
    )

    axstate = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[sub], hspace=0.2)

    ax1 = fig.add_subplot(axstate[1:])
    sess.replay.expvar.plot(ax=ax1, tstart=post[0])
    # if sub == 0:
    #     ax1.set_ylim([0, 0.55])
    # else:
    #     ax1.set_ylim([0, 0.4])

    # if i > 0:
    #     ax1.spines["left"].set_visible(False)
    #     ax1.set_yticks([])
    #     ax1.set_yticklabels([])
    #     ax1.set_ylabel("")
    #     ax1.legend("")
    #     ax1.set_title("")

    axhypno = fig.add_subplot(axstate[0], sharex=ax1)
    sess.brainstates.hypnogram(ax1=axhypno, tstart=post[0], unit="h")
    axhypno.set_title(sess.sessinfo.session.sessionName)
    # panel_label(axhypno, "a")
    # ax1.set_ylim([0, 0.3])


# endregion

