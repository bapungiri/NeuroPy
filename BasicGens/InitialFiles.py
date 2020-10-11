#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter

from callfunc import processData

basePath = [
    # "/data/Clustering/SleepDeprivation/RatJ/Day1/",
    # "/data/Clustering/SleepDeprivation/RatK/Day1/",
    # "/data/Clustering/SleepDeprivation/RatN/Day1/",
    # "/data/Clustering/SleepDeprivation/RatJ/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day2/",
    # "/data/Clustering/SleepDeprivation/RatN/Day2/",
    # "/data/Clustering/SleepDeprivation/RatJ/Day4/",
    # "/data/Clustering/SleepDeprivation/RatK/Day4/",
    # "/data/Clustering/SleepDeprivation/RatN/Day4/",
    "/data/Clustering/SleepDeprivation/RatA14d1LP/Rollipram/",
]


sessions = [processData(_) for _ in basePath]


#%% Generate _basics.npy files
# region

for sub, sess in enumerate(sessions):

    # sess.trange = np.array([])
    # badchans = [0]
    sess.recinfo.makerecinfo()
# endregion

#%% Generate .prb for spyking circus
# region
for sub, sess in enumerate(sessions):

    sess.trange = np.array([])
    sess.makePrmPrb.makePrbCircus(probetype="diagbio", shanksCombine=1)

# endregion

#%% artifacts file gen
# region
for sub, sess in enumerate(sessions):

    sess.trange = np.array([])
    # sess.artifact.usingZscore(thresh=10)
    sess.artifact.plot(chan=32)
# endregion

#%% Generate position files
# region
for sub, sess in enumerate(sessions):
    sess.trange = np.array([])
    sess.position.getPosition()
    # sess.position.export2Neuroscope()

# endregion

#%% epochs from csv
# region
for sub, sess in enumerate(sessions):
    sess.trange = np.array([])
    sess.epochs.getfromCSV()
# endregion

#%% Gen instantenous firing rate
# region

for sub, sess in enumerate(sessions):
    sess.trange = np.array([])
    sess.spikes.gen_instfiring()
# endregion

#%%Generate bestThetachan
# region

for sub, sess in enumerate(sessions):
    sess.trange = np.array([])
    sess.theta.detectBestChan()
# endregion

