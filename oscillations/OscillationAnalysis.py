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
    "/data/Clustering/SleepDeprivation/RatN/Day1/",
    # "/data/Clustering/SleepDeprivation/RatJ/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day2/",
    # "/data/Clustering/SleepDeprivation/RatN/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day4/"
]


sessions = [processData(_) for _ in basePath]


#%% Bicoherence analysis on ripples
# region

colmap = Colormap().dynamicMap()

for sub, sess in enumerate(sessions):

    sess.trange = np.array([])
    eegSrate = sess.recinfo.lfpSrate
    ripples = sess.ripple.time
    lfp = sess.spindle.best_chan_lfp()[0]

    lfpripple = []
    for ripple in ripples:
        start = int(ripple[0] * eegSrate)
        end = int(ripple[1] * eegSrate)
        lfpripple.extend(lfp[start:end])

    lfpripple = np.asarray(lfpripple)
    bicoh, freq, bispec = signal_process.bicoherence(lfpripple, fhigh=300)

    bicohsmth = gaussian_filter(bicoh, sigma=3)
    # bicoh = np.where(bicoh > 0.05, bicoh, 0)
    plt.clf()
    fig = plt.figure(1, figsize=(10, 15))
    gs = gridspec.GridSpec(1, 2, figure=fig)
    fig.subplots_adjust(hspace=0.3)
    ax = fig.add_subplot(gs[0, 1])
    ax.clear()
    im = ax.pcolorfast(
        freq, freq, np.sqrt(bicohsmth), cmap=colmap, vmax=0.5, vmin=0.018
    )
    # ax.contour(freq, freq, bicoh, levels=[0.1, 0.2, 0.3], colors="k", linewidths=1)
    ax.set_ylim([1, max(freq) / 2])
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Frequency (Hz)")

    # cax = fig.add_axes([0.3, 0.8, 0.5, 0.05])
    # cax.clear()
    # ax.contour(freq, freq, bicoh, levels=[0.1, 0.2, 0.3], colors="k", linewidths=1)
    fig.colorbar(im, ax=ax, orientation="horizontal")

# endregion
fundamental
