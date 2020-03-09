import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sessionObj import session

# from lfpEvent import swr

# alt.renderers.enable("html")

# alt.data_transformers.enable("json")

basePath = [
    # "/data/Clustering/SleepDeprivation/RatJ/Day1/",
    # "/data/Clustering/SleepDeprivation/RatJ/Day2/",
    # # "/data/Clustering/SleepDeprivation/RatJ/Day3/",
    # # "/data/Clustering/SleepDeprivation/RatJ/Day4/",
    "/data/Clustering/SleepDeprivation/RatK/Day1/",
    # "/data/Clustering/SleepDeprivation/RatK/Day2/",
    # "/data/Clustering/SleepDeprivation/RatN/Day1/",
    # "/data/Clustering/SleepDeprivation/RatN/Day2/",
]

sessions = [session(_) for _ in basePath]


for sess_id, sess in enumerate(sessions):
    # sess.detect_hswa()
    # sess.findswr()
    # sess.load_swr_evt()
    sess.plot_psth_hswa_ripple()


# plt.clf()
# for sess_id, sess in enumerate(sessions, 1):
#     pk_power = sess.ripples["peakPower"]

#     plt.subplot(3, 2, sess_id)
#     plt.plot(pk_power)

