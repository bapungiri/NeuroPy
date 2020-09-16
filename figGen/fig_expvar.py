#%% Imports
import warnings
import os
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.pyplot import title
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from scipy.ndimage import gaussian_filter

from callfunc import processData
from plotUtil import savefig

warnings.simplefilter(action="default")

basePath = [
    "/data/Clustering/SleepDeprivation/RatJ/Day1/",
    "/data/Clustering/SleepDeprivation/RatK/Day1/",
    "/data/Clustering/SleepDeprivation/RatN/Day1/",
    "/data/Clustering/SleepDeprivation/RatJ/Day2/",
    "/data/Clustering/SleepDeprivation/RatK/Day2/",
    "/data/Clustering/SleepDeprivation/RatN/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day4/",
    # "/data/Clustering/SleepDeprivation/RatN/Day4/",
]


sessions = [processData(_) for _ in basePath]

plt.clf()
fig = plt.figure(1, figsize=(8.5, 11))
gs = gridspec.GridSpec(5, 5, figure=fig)
fig.subplots_adjust(hspace=0.4, wspace=0.5)
fig.suptitle("Assessing Replay")
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
evsd = []
for sub, sess in enumerate(sessions[1:3]):

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

    ev = pd.DataFrame(
        {
            "time": (sess.replay.expvar.t_match - post[0]) / 3600,
            "expvar": sess.replay.expvar.ev.squeeze().mean(axis=0),
            "rev": sess.replay.expvar.rev.squeeze().mean(axis=0),
        }
    )
    evsd.append(ev)

conf_interval = int(68.2 / np.sqrt(len(evsd)))  # SEM, standard deviation = 68.2
evsd = pd.concat(evsd)
ax = plt.subplot(gs[0, :2])
ax.clear()
sns.lineplot(
    data=evsd,
    x="time",
    y="rev",
    ci=conf_interval,
    ax=ax,
    legend=None,
    color="green",
    n_boot=10,
    seed=10,
)
sns.lineplot(
    data=evsd,
    x="time",
    y="expvar",
    ci=conf_interval,
    color="black",
    ax=ax,
    legend=None,
    n_boot=10,
    seed=10,
)
ax.set_ylabel("Replay")
ax.set_xlabel("Time (h)")
ax.set_title("Replay during sleep deprivaiton", fontsize=titlesize)
ax.legend(["REV", "EV"])
panel_label(ax, "a")

# endregion

#%% Explained variance for control sessions

# region
"""Only found stable units for 3 sessions
"""
evsd = []
for sub, sess in enumerate(sessions[5:6]):

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

    ev = pd.DataFrame(
        {
            "time": (sess.replay.expvar.t_match - post[0]) / 3600,
            "expvar": sess.replay.expvar.ev.squeeze().mean(axis=0),
            "rev": sess.replay.expvar.rev.squeeze().mean(axis=0),
        }
    )
    evsd.append(ev)

evsd = pd.concat(evsd)
ax = plt.subplot(gs[0, 2:4])
ax.clear()
sns.lineplot(data=evsd, x="time", y="rev", ci=68, ax=ax, legend=None, color="green")
sns.lineplot(
    data=evsd, x="time", y="expvar", ci=68, color="black", ax=ax, legend=None,
)
ax.set_ylabel("Replay")
ax.set_xlabel("Time (h)")
ax.set_title("Replay during normal sleep", fontsize=titlesize)
panel_label(ax, "b")
# ax.legend(["REV", "EV"])

# endregion

#%% Replay during recovery sleep (template=ZT5, control=ZT1)

# region
"""Only found stable units for 3 sessions
"""
evsd = []
for sub, sess in enumerate(sessions[1:3]):

    sess.trange = np.array([])
    maze = sess.epochs.maze
    post = sess.epochs.post
    pre = sess.epochs.pre

    bins = [
        [maze[0], maze[1]],
        # [post[0] + 1 * 3600, post[0] + 2 * 3600],
        # [post[0] + 2 * 3600, post[0] + 3 * 3600],
        # [post[0] + 3 * 3600, post[0] + 4 * 3600],
        [post[0] + 4 * 3600, post[0] + 5 * 3600],
        [post[0] + 5 * 3600, post[0] + 6 * 3600],
        [post[0] + 6 * 3600, post[0] + 7 * 3600],
    ]
    sess.spikes.stability.firingRate(bins=bins)
    sess.replay.expvar.compute(
        template=bins[1],
        match=[post[0] + 5 * 3600, post[0] + 7 * 3600],
        control=bins[0],
    )

    ev = pd.DataFrame(
        {
            "time": (sess.replay.expvar.t_match - (post[0] + 5 * 3600)) / 3600,
            "expvar": sess.replay.expvar.ev.squeeze().mean(axis=0),
            "rev": sess.replay.expvar.rev.squeeze().mean(axis=0),
        }
    )
    evsd.append(ev)

conf_interval = int(68.2 / np.sqrt(len(evsd)))  # SEM, standard deviation = 68.2
evsd = pd.concat(evsd)

ax = plt.subplot(gs[1, :2])
ax.clear()
sns.lineplot(
    data=evsd,
    x="time",
    y="rev",
    ci=conf_interval,
    ax=ax,
    legend=None,
    color="green",
    n_boot=10,
    seed=10,
)
sns.lineplot(
    data=evsd,
    x="time",
    y="expvar",
    ci=conf_interval,
    color="black",
    ax=ax,
    legend=None,
    n_boot=10,
    seed=10,
)
ax.set_ylabel("Replay")
ax.set_xlabel("Time (h)")
ax.set_title("Replay during recovery sleep", fontsize=titlesize)
panel_label(ax, "c")
# ax.legend(["REV", "EV"])

# endregion


#%% ICA reactivation during sleep deprivation
# region

# endregion


scriptname = os.path.basename(__file__)
folder = "/home/bapung/Documents/MATLAB/figures/compileFigures/figures"
filename = "fig_Replay1"
savefig(fig, filename, scriptname, folder=folder)
