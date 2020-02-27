import numpy as np
import matplotlib.pyplot as plt
from pfPlot import pf
import pandas as pd
import seaborn as sns

import altair as alt

# from bokeh.plotting import figure, save, output_file, show
# from bokeh.models import Legend, LegendItem
# from bokeh.layouts import row, column, grid, gridplot, layout

# output_file("pf.html")

basePath = [
    "/data/Clustering/SleepDeprivation/RatJ/Day1/",
    # "/data/Clustering/SleepDeprivation/RatJ/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day1/",
    # "/data/Clustering/SleepDeprivation/RatK/Day2/",
    # "/data/Clustering/SleepDeprivation/RatN/Day1/",
    # "/data/Clustering/SleepDeprivation/RatN/Day2/",
]

sessions = [pf(x) for x in basePath]


for i, sess in enumerate(sessions):
    sess.pf1d()


plt.clf()


for i, (t, y) in enumerate(zip(sess.spkt, sess.spky)):

    plt.subplot(6, 8, i + 1)
    plt.plot(sess.y, sess.t, color="gray", alpha=0.5, linewidth=1)
    plt.plot(y, t, "r.", markersize=2)
    plt.axis("off")


# ax = plt.plot(sess.t[:-1], sess.speed, color="navy", alpha=0.5)


# ax = plt.plot(sess.t, sess.y, color="navy", alpha=0.5)

# plt.plot(sess.x, sess.y, color="navy", alpha=0.5)
# save(p2)
