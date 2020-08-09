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
from signal_process import spectrogramBands
import sys


plt.clf()

basePath = [
    # "/data/Clustering/SleepDeprivation/RatJ/Day1/",
    # "/data/Clustering/SleepDeprivation/RatK/Day1/",
    # "/data/Clustering/SleepDeprivation/RatN/Day1/",
    # "/data/Clustering/SleepDeprivation/RatJ/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day2/",
    # "/data/Clustering/SleepDeprivation/RatN/Day2/",
    "/data/Clustering/SleepDeprivation/RatN/Day4/",
]


sessions = [processData(_) for _ in basePath]


#%% visualize detection
for sub, sess in enumerate(sessions):

    # sess.recinfo.makerecinfo()
    sess.trange = np.array([])
    # sess.theta.detectBestChan()
    sess.brainstates.detect()
    # sess.brainstates.plot()


# a = sess.spindle.best_chan_lfp()[0]
# spec = spectrogramBands(a)

# fig = plt.figure(1, figsize=(6, 10))
# gs = GridSpec(4, 1, figure=fig)
# fig.subplots_adjust(hspace=0.4)
# fig.canvas.mpl_connect("key_press_event", press)

# ax = fig.add_subplot(gs[1, :])
# sxx = spec.sxx / np.max(spec.sxx)
# sxx = gaussian_filter(sxx, sigma=1)
# print(np.max(sxx), np.min(sxx))
# vmax = np.max(sxx) / 1000
# g = ax.pcolorfast(spec.time, spec.freq, sxx, cmap="YlGn", vmax=vmax)
# ax.set_ylim([0, 60])
# ax.set_xlim([np.min(spec.time), np.max(spec.time)])
# ax.set_xlabel("Time (s)")
# ax.set_ylabel("Frequency (s)")
