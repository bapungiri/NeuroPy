#%%
from collections import namedtuple
from pathlib import Path

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns

import signal_process
from callfunc import processData
from plotUtil import Colormap


#%% Subjects
basePath = [
    # "/data/Clustering/SleepDeprivation/RatJ/Day1/",
    # "/data/Clustering/SleepDeprivation/RatK/Day1/",
    # "/data/Clustering/SleepDeprivation/RatN/Day1/",
    # "/data/Clustering/SleepDeprivation/RatJ/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day2/",
    "/data/Clustering/SleepDeprivation/RatN/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day4/"
    # "/data/Clustering/SleepDeprivation/RatN/Day4/"
    # "/data/Clustering/SleepDeprivation/RatA14d1LP/Rollipram/",
]


sessions = [processData(_) for _ in basePath]


#%% Ripples
# region
# plt.clf()

# fig, ax = plt.subplots(figsize=(5, 3))
for sub, sess in enumerate(sessions):

    sess.trange = np.array([])
    
    
    sess.recinfo.
    # sess.ripple.channels()
    # sess.ripple.detect()
    # sess.ripple.export2Neuroscope()
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
    sess.spindle.channels()
    sess.spindle.detect()
    # sess.spindle.plot()

# ax.set_xlim([-5, 10])
# endregion

#%% H-SWA
# region
plt.clf()
for sub, sess in enumerate(sessions):

    sess.trange = np.array([])

    sess.swa.detect()
    # sess.swa.plot()
# endregion

#%% Best Theta channel based on
# region
for sub, sess in enumerate(sessions):

    sess.trange = np.array([])
    # sess.theta.detectBestThetaChan()
# endregion
