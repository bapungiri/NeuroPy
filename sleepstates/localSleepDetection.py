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
plt.clf()
fig = plt.figure(1, figsize=(1, 15))
gs = GridSpec(4, 3, figure=fig)
fig.subplots_adjust(hspace=0.5)


# p = Pac(idpac=(6, 3, 0), f_pha=(4, 10, 1, 1), f_amp=(30, 100, 5, 5))
sigma = 25
t_gauss = np.arange(-1, 1, 0.001)
A = 1 / np.sqrt(2 * np.pi * sigma ** 2)
gaussian = A * np.exp(-(t_gauss ** 2) / (2 * sigma ** 2))

for sub, sess in enumerate(sessions):

    sess.trange = np.array([])
    tstart = sess.epochs.post[0]
    tend = sess.epochs.post[0] + 5 * 3600
    sess.spikes.fromCircus("diff_folder")
    spikes = sess.spikes.times

    lfp, _, _ = sess.spindle.best_chan_lfp()
    t = np.linspace(0, len(lfp) / 1250, len(lfp))

    tstart = sess.epochs.post[0]
    tend = sess.epochs.post[0] + 5 * 3600

    lfpsd = stats.zscore(lfp[(t > tstart) & (t < tend)]) + 80

    tbin = np.arange(tstart, tend, 0.001)
    mua = np.concatenate(spikes)

    spikecount = np.histogram(mua, tbin)[0]

    instfiring = sg.convolve(spikecount, gaussian, mode="same", method="direct")

    off = np.diff(np.where(instfiring < np.median(instfiring), 1, 0))
    start_ripple = np.where(off == 1)[0]
    stop_ripple = np.where(off == -1)[0]

    if start_ripple[0] > stop_ripple[0]:
        stop_ripple = stop_ripple[1:]
    if start_ripple[-1] > stop_ripple[-1]:
        start_ripple = start_ripple[:-1]

    offperiods = np.vstack((start_ripple, stop_ripple)).T
    duration = np.diff(offperiods, axis=1)
    quantiles = pd.qcut(duration.squeeze(), 10, labels=False)

    top10percent = np.where(quantiles == 9)[0]

    # offperiods = offperiods[top10percent, :]
    for (start, end) in offperiods:
        plt.plot([tbin[start], tbin[start]], [0, 300], "r")
        plt.plot([tbin[end], tbin[end]], [0, 300], "k")

    plt.plot(np.linspace(tstart, tend, len(lfpsd)), lfpsd, "k")
    for cell, spk in enumerate(spikes):
        spk = spk[(spk > tstart) & (spk < tend)]
        plt.plot(spk, cell * np.ones(len(spk)), "|")

    # per_min_periods = np.arange(tstart, tend, 3600)
    # for beg in per_min_periods:

    tbin_offperiods = np.linspace(tstart, tend, 6)
    t_offperiods = tbin[offperiods[:, 0]]
    hist_off = np.histogram(t_offperiods, bins=tbin_offperiods)[0]
