#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib
from collections import namedtuple
from pathlib import Path
import matplotlib.gridspec as gridspec
import signal_process
import matplotlib as mpl
from plotUtil import Colormap

cmap = matplotlib.cm.get_cmap("hot_r")


from callfunc import processData

#%% Subjects
basePath = [
    # "/data/Clustering/SleepDeprivation/RatJ/Day1/",
    # "/data/Clustering/SleepDeprivation/RatK/Day1/",
    # "/data/Clustering/SleepDeprivation/RatN/Day1/",
    # "/data/Clustering/SleepDeprivation/RatJ/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day2/",
    # "/data/Clustering/SleepDeprivation/RatN/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day4/"
    "/data/Clustering/SleepDeprivation/RatN/Day4/"
]


sessions = [processData(_) for _ in basePath]


#%% Ripples
# region
plt.clf()

# fig, ax = plt.subplots(figsize=(5, 3))
for sub, sess in enumerate(sessions):

    sess.trange = np.array([])
    sess.ripple.channels()
    sess.ripple.detect()
    # sess.ripple.plot()
    # sess.spindle.channels()
    # sess.spindle.detect()
    # sess.spindle.plot()
    # sess.swa.detect()
    # sess.swa.plot()
    # _, b, c = sess.ripple.best_chan_lfp()

# ax.set_xlim([-5, 10])
# endregion

#%% Spindles
# region
plt.clf()

for sub, sess in enumerate(sessions):

    sess.trange = np.array([])
    # sess.spindle.channels()
    # sess.spindle.detect()
    sess.spindle.plot()

# ax.set_xlim([-5, 10])
# endregion

#%% H-SWA
# region
plt.clf()
for sub, sess in enumerate(sessions):

    sess.trange = np.array([])

    sess.swa.detect()
    sess.swa.plot()
# endregion

#%% Best Theta channel based on
# region
for sub, sess in enumerate(sessions):

    sess.trange = np.array([])
    # sess.theta.detectBestThetaChan()
# endregion

