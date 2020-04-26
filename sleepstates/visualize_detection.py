import numpy as np
from callfunc import processData
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.cluster import KMeans
from hmmlearn.hmm import GaussianHMM
import scipy.stats as stats
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter


def make_boxes(
    ax, xdata, ydata, xerror, yerror, facecolor="r", edgecolor="None", alpha=0.5
):

    # Loop over data points; create box from errors at each point
    errorboxes = [
        Rectangle((x, y), xe, ye) for x, y, xe, ye in zip(xdata, ydata, xerror, yerror)
    ]

    # Create patch collection with specified colour/alpha
    pc = PatchCollection(
        errorboxes, facecolor=facecolor, alpha=alpha, edgecolor=edgecolor
    )

    # Add collection to axes
    ax.add_collection(pc)

    # Plot errorbars
    # artists = ax.errorbar(
    #     xdata, ydata, xerr=xerror, yerr=yerror, fmt="None", ecolor="k"
    # )
    return 1


basePath = [
    # "/data/Clustering/SleepDeprivation/RatJ/Day1/",
    # "/data/Clustering/SleepDeprivation/RatK/Day1/",
    "/data/Clustering/SleepDeprivation/RatN/Day1/",
    # "/data/Clustering/SleepDeprivation/RatJ/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day2/",
    # "/data/Clustering/SleepDeprivation/RatN/Day2/",
]


sessions = [processData(_) for _ in basePath]


for sub, sess in enumerate(sessions):

    # sess.recinfo.makerecinfo()
    sess.trange = np.array([])
    sess.brainstates.detect()


sxx = sess.brainstates.sxx[0:40, :-1]
sxx = gaussian_filter(sxx, sigma=1)

df = sess.brainstates.params
states = sess.brainstates.states
t = df["time"]
freq = np.linspace(0, 38, sxx.shape[0] + 1)

# #7bbcb6plt.close("all")
plt.clf()
fig = plt.figure(1, figsize=(6, 10))
gs = GridSpec(9, 1, figure=fig)
fig.subplots_adjust(hspace=0.4)

ax1 = fig.add_subplot(gs[1:3, 0])
# ax1.imshow(
#     sxx,
#     cmap="YlGn",
#     aspect="auto",
#     # extent=[0, max(t), 0, 40.0],
#     origin="lower",
#     vmin=-0.1,
#     vmax=7,
#     # interpolation="mitchell",
# )

# ax1.set_xticks(t)
# x_label_list = [str(_) for _ in t]
# ax1.set_xticks(np.arange(0, sxx.shape[0]))

# ax1.set_xticklabels(x_label_list)


ax1.pcolorfast(
    t,
    freq,
    sxx,
    cmap="YlGn",
    vmin=-0.1,
    vmax=6,
    # interpolation="mitchell",
)
ax1.set_ylabel("Frequency(Hz)")

ax2 = fig.add_subplot(gs[3, 0], sharex=ax1)
ax2.plot(df["time"], df["emg"], color="#ea7b7b")
ax2.set_ylabel("Emg")


# ax3 = fig.add_subplot(gs[3, 0], sharex=ax1)
# ax3.plot(df["time"], df["theta"])
# ax3.set_ylabel("Theta")


# ax4 = fig.add_subplot(gs[4, 0], sharex=ax1)
# ax4.plot(df["time"], df["delta"])
# ax4.set_ylim(-2, 5)
# ax4.set_ylabel("Delta")


ax5 = fig.add_subplot(gs[4, 0], sharex=ax1)
ax5.plot(df["time"], df["theta_delta_ratio"], color="#7bbcb6")
ax5.set_ylabel("theta/delta")
ax5.set_xlabel("Time (s)")


ax6 = fig.add_subplot(gs[0, 0], sharex=ax1)

# for i in [1, 2, 3, 4]:
# fig, ax = plt.subplots(1)
x = np.asarray(states.start)
y = np.zeros(len(x)) + np.asarray(states.state)
width = np.asarray(states.duration)
height = np.ones(len(x))
qual = states.state

colors = ["#6b90d1", "#eb9494", "#b6afaf", "#474343"]
col = [colors[int(state) - 1] for state in states.state]

make_boxes(ax6, x, y, width, height, facecolor=col)
# ax6.set_xlim(0, 50000)
ax6.set_ylim(1, 5)
ax6.annotate("wake", (-0.8, 4.5), transform=ax6.transAxes)

# states = df["state"]
# x = np.arange(0, len(states))

# nrem = np.where(states == 1, 1, 0)
# ax6.fill_between(x, nrem, 0, color="#6b90d1")

# rem = np.where(states == 2, 1, 0)
# ax6.fill_between(x, rem, 0, color="#eb9494")

# qw = np.where(states == 3, 1, 0)
# ax6.fill_between(x, qw, 0, color="#b6afaf")

# active = np.where(states == 4, 1, 0)
# ax6.fill_between(x, active, 0, color="#474343")
# ax6.legend(("nrem", "rem", "quiet", "wake"), loc=2)
ax6.axis("off")

# ax7 = fig.add_subplot(gs[6, 0], sharex=ax1)
# ax7.plot(df["time"], df["ripple"])
# ax7.set_ylabel("ripple")

# ax8 = fig.add_subplot(gs[7, 0], sharex=ax1)
# ax8.plot(df["time"], df["spindle"])
# ax8.set_ylabel("spindle")

# ax9 = fig.add_subplot(gs[8, 0], sharex=ax1)
# ax9.plot(df["time"], df["gamma"])
# ax9.set_ylabel("gamma")
# ax9.set_xlabel("Time (s)")
# ax6.legend(['df','sdf','df','sdf'])
ax6.annotate("quiet", (-0.1, 3.5))
ax6.annotate("rem", (0.1, 2.5))
ax6.annotate("nrem", (-10, 1.5))

# fig.text(5, 6, "fsfffffg")
