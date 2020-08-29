import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter1d

from callfunc import processData

basePath = [
    # "/data/Clustering/SleepDeprivation/RatJ/Day1/",
    # "/data/Clustering/SleepDeprivation/RatK/Day1/",
    # "/data/Clustering/SleepDeprivation/RatN/Day1/",
    # "/data/Clustering/SleepDeprivation/RatJ/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day2/",
    # "/data/Clustering/SleepDeprivation/RatN/Day2/"
    # "/data/Clustering/SleepDeprivation/RatK/Day4/"
    "/data/Clustering/SleepDeprivation/RatN/Day4/"
]


sessions = [processData(_) for _ in basePath]

#%% Bayesian decoding in open field
# region
plt.close("all")
for sub, sess in enumerate(sessions):

    sess.trange = np.array([])
    sess.decode.bayes2d.fit()
    sess.decode.bayes2d.plot()


# endregion

#%% Decoding population burst events during MAZE of open field env
# region
for sub, sess in enumerate(sessions):
    sess.trange = np.array([])
    maze = sess.epochs.maze
    pbe = sess.pbe.events
    maze_pbe = pbe[(pbe.start > maze[0]) & (pbe.start < maze[1])]
    decodedPos = sess.decode.bayes2d.decode(maze_pbe, binsize=0.02)


# endregion
