import numpy as np
from callfunc import processData
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.cluster import KMeans
from hmmlearn.hmm import GaussianHMM
import scipy.stats as stats
import pandas as pd


basePath = [
    "/data/Clustering/SleepDeprivation/RatJ/Day1/",
    # "/data/Clustering/SleepDeprivation/RatK/Day1/",
    # "/data/Clustering/SleepDeprivation/RatN/Day1/",
    # "/data/Clustering/SleepDeprivation/RatJ/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day2/",
    # "/data/Clustering/SleepDeprivation/RatN/Day2/"
]


sessions = [processData(_) for _ in basePath]


for sub, sess in enumerate(sessions):

    # sess.recinfo.makerecinfo()
    sess.trange = np.array([])
    sess.brainstates.detect()


df = [
    sess.brainstates.params_pre,
    sess.brainstates.params_maze,
    sess.brainstates.params_post,
]

df = pd.concat(df, ignore_index=True)

sxx = np.concatenate(
    (sess.brainstates.sxx_pre, sess.brainstates.sxx_maze, sess.brainstates.sxx_post),
    axis=1,
)

sxx = stats.zscore(sxx[0:20, :], axis=0)

# sess.brainstates.pre_params.hist(bins=400)
# sess.brainstates.pre_params.plot(y="emg")
plt.close("all")
plt.clf()
fig = plt.figure(1, figsize=(6, 10))
gs = GridSpec(9, 1, figure=fig)
fig.subplots_adjust(hspace=0.4)

ax1 = fig.add_subplot(gs[1, 0])
ax1.imshow(
    sxx[:, :],
    cmap="YlGn",
    aspect="auto",
    # extent=[0, max(t) / 3600, 0, 30.0],
    origin="lower",
    vmin=-0.1,
    vmax=3,
    # interpolation="mitchell",
)
ax1.set_ylabel("Frequency(Hz)")

ax2 = fig.add_subplot(gs[2, 0], sharex=ax1)
ax2.plot(df["emg"])
ax2.set_ylabel("Emg")


ax3 = fig.add_subplot(gs[3, 0], sharex=ax1)
ax3.plot(df["theta"])
ax3.set_ylabel("Theta")


ax4 = fig.add_subplot(gs[4, 0], sharex=ax1)
ax4.plot(df["delta"])
ax4.set_ylim(-2, 5)
ax4.set_ylabel("Delta")


ax5 = fig.add_subplot(gs[5, 0], sharex=ax1)
ax5.plot(df["theta_delta_ratio"])
ax5.set_ylabel("theta/delta")
ax5.set_xlabel("Time (s)")


ax6 = fig.add_subplot(gs[0, 0], sharex=ax1)

states = df["state"]
x = np.arange(0, len(states))

nrem = np.where(states == 1, 1, 0)
ax6.fill_between(x, nrem, 0, color="#6b90d1")

rem = np.where(states == 2, 1, 0)
ax6.fill_between(x, rem, 0, color="#eb9494")

qw = np.where(states == 3, 1, 0)
ax6.fill_between(x, qw, 0, color="#b6afaf")

active = np.where(states == 4, 1, 0)
ax6.fill_between(x, active, 0, color="#201d1d")
ax6.legend(("nrem", "rem", "quiet", "wake"), loc=2)
ax6.axis("off")

ax7 = fig.add_subplot(gs[6, 0], sharex=ax1)
ax7.plot(df["ripple"])
ax7.set_ylabel("ripple")

ax8 = fig.add_subplot(gs[7, 0], sharex=ax1)
ax8.plot(df["spindle"])
ax8.set_ylabel("spindle")

ax9 = fig.add_subplot(gs[8, 0], sharex=ax1)
ax9.plot(df["gamma"])
ax9.set_ylabel("gamma")
ax9.set_xlabel("Time (s)")
