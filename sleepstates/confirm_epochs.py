import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter

from callfunc import processData
from signal_process import spectrogramBands

basePath = [
    # "/data/Clustering/SleepDeprivation/RatJ/Day1/",
    # "/data/Clustering/SleepDeprivation/RatK/Day1/",
    # "/data/Clustering/SleepDeprivation/RatN/Day1/",
    # "/data/Clustering/SleepDeprivation/RatJ/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day2/",
    "/data/Clustering/SleepDeprivation/RatN/Day2/"
]
sess = processData(basePath[0])
file = sess.sessinfo.recfiles.eegfile
eeg = np.memmap(file, dtype="int16", mode="r")
eeg = np.memmap.reshape(eeg, (int(len(eeg) / 134), 134))

eegchan = eeg[:, 42]

obj = spectrogramBands(eegchan)

theta = np.load(sess.sessinfo.files.thetalfp)
obj2 = spectrogramBands(theta)


plt.clf()
fig = plt.figure(1, figsize=(6, 10))
gs = GridSpec(9, 1, figure=fig)
fig.subplots_adjust(hspace=0.4)

ax1 = fig.add_subplot(gs[1:3, 0])

ax1.pcolorfast(
    obj.time,
    obj.freq,
    obj.sxx,
    cmap="YlGn",
    vmin=-0.1,
    vmax=10000,
    # interpolation="mitchell",
)
ax2 = fig.add_subplot(gs[4, 0])

ax2.pcolorfast(
    obj2.time,
    obj2.freq,
    obj2.sxx,
    cmap="YlGn",
    vmin=-0.1,
    vmax=10000,
    # interpolation="mitchell",
)
