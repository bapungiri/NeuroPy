import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib


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
    "/data/Clustering/SleepDeprivation/RatK/Day1/",
    # "/data/Clustering/SleepDeprivation/RatK/Day2/",
    # "/data/Clustering/SleepDeprivation/RatN/Day1/",
    # "/data/Clustering/SleepDeprivation/RatN/Day2/"
]

sessions = [processData(_) for _ in basePath]


plt.close()
fig = []
for sess in sessions:
    # sess.recinfo.makerecinfo()
    # sess.spksrt_param.makePrbCircus(probetype="diagbio")
    sig_zscore = sess.artifact.usingZscore()
    # sess.artifact.plot
    plt.plot(sig_zscore)
    # sess.eventpsth.hswa_ripple.nQuantiles = 5
    # temp = sess.eventpsth.hswa_ripple.plot()
    # fig.append(temp)

plt.show()
# for f in fig:
#     f.show()
