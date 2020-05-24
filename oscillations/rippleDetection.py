import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib
from collections import namedtuple

cmap = matplotlib.cm.get_cmap("hot_r")


from callfunc import processData

basePath = [
    # "/data/Clustering/SleepDeprivation/RatJ/Day1/",
    # "/data/Clustering/SleepDeprivation/RatK/Day1/",
    "/data/Clustering/SleepDeprivation/RatN/Day1/",
    # "/data/Clustering/SleepDeprivation/RatJ/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day2/",
    # "/data/Clustering/SleepDeprivation/RatN/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day4/"
]


sessions = [processData(_) for _ in basePath]


plt.clf()
group = []
for sub, sess in enumerate(sessions):

    sess.trange = np.array([])
    # sess.ripple.channels()
    # sess.ripple.detect()

    # sess.ripple.plot()
    # a, _, _ = sess.ripple.best_chan_lf
