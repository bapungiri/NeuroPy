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
    # sess.brainstates.detect()


states = sess.brainstates.states

plt.clf()
fig = plt.figure(1, figsize=(6, 10))
gs = GridSpec(9, 1, figure=fig)
fig.subplots_adjust(hspace=0.4)


ax = fig.add_subplot(gs[0, 0])

# for i in [1, 2, 3, 4]:
# fig, ax = plt.subplots(1)
x = np.asarray(states.start)
y = np.zeros(len(x)) + np.asarray(states.state)
width = np.asarray(states.duration)
height = np.ones(len(x))
qual = states.state

colors = ["#6b90d1", "#eb9494", "#b6afaf", "#474343"]
col = [colors[int(state) - 1] for state in states.state]

make_boxes(ax, x, y, width, height, facecolor=col)
ax.set_xlim(0, 50000)
ax.set_ylim(1, 5)
