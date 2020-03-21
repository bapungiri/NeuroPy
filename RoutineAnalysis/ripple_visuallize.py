import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib
import plotting as pt

b = pt.check.a(3)

mpl.style.use("figPublish")

# plt.style.use("figPublish")
# matplotlib.use("Qt5Agg")
from callfunc import processData

# TODO thoughts on using data class for loading data into function

basePath = [
    # "/data/Clustering/SleepDeprivation/RatJ/Day1/",
    # "/data/Clustering/SleepDeprivation/RatJ/Day2/",
    # "/data/Clustering/SleepDeprivation/RatJ/Day3/",
    # "/data/Clustering/SleepDeprivation/RatJ/Day4/",
    # "/data/Clustering/SleepDeprivation/RatK/Day1/",
    # "/data/Clustering/SleepDeprivation/RatK/Day2/",
    "/data/Clustering/SleepDeprivation/RatN/Day1/",
    "/data/Clustering/SleepDeprivation/RatN/Day2/",
]

sessions = [processData(_) for _ in basePath]

plt.close()
fig = []
for sess in sessions:
    sess.eventpsth.hswa_ripple.nQuantiles = 5
    temp = sess.eventpsth.hswa_ripple.plot()
    fig.append(temp)

    # for f in fig:
# fig.axes[0].set_facecolor("white")
# fig.axes[0].spines["left"].set_visible(False)
# fig.axes[0].spines["bottom"].set_visible(True)
# fig.show()
# with plt.style.context("figPublish"):
# fig.draw
for f in fig:
    f.show()
# fig.axes.spines["left"].set_visible(False)
