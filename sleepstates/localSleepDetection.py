import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.signal as sg
from matplotlib.gridspec import GridSpec
import seaborn as sns
from signal_process import filter_sig, hilbertfast
from statsmodels.tsa.stattools import grangercausalitytests
import matplotlib as mpl
from scipy.ndimage import gaussian_filter1d

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

# during REM sleep
plt.close("all")
# fig = plt.figure(1, figsize=(1, 15))
# gs = GridSpec(4, 3, figure=fig)
# fig.subplots_adjust(hspace=0.5)


# p = Pac(idpac=(6, 3, 0), f_pha=(4, 10, 1, 1), f_amp=(30, 100, 5, 5))
sigma = 0.025
t_gauss = np.arange(-1, 1, 0.001)
A = 1 / np.sqrt(2 * np.pi * sigma ** 2)
gaussian = A * np.exp(-(t_gauss ** 2) / (2 * sigma ** 2))

for sub, sess in enumerate(sessions):

    sess.trange = np.array([])
    tstart = sess.epochs.post[0]
    tend = sess.epochs.post[0] + 5 * 3600
    sess.spikes.fromCircus("diff_folder")

    # sess.localsleep.detect(period=[tstart, tend])

    fig = sess.localsleep.plot()

    # tbin_offperiods = np.linspace(tstart, tend, 6)
    # t_offperiods = tbin[offperiods[:, 0]]
    # hist_off = np
    # .histogram(t_offperiods, bins=tbin_offperiods)[0]


# ax = fig.add_subplot(358)
# ax.plot(np.arange(10))
