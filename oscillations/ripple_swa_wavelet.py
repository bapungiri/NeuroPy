import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib
from collections import namedtuple
from signal_process import wavelet_decomp, filter_sig
import time
import progressbar

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
    ripples = sess.ripple.time
    a, chans, _ = sess.ripple.best_chan_lfp()


duration = np.diff(ripples, axis=1)
peakpower = sess.ripple.peakpower
longrpl = np.where(peakpower > 20)[0]
ripples = np.delete(ripples, longrpl, 0)

lfp = stats.zscore(a[1, :])

# with progressbar.ProgressBar(max_value=len(ripples)) as bar:
# frequency = np.logspace(np.log10(100), np.log10(250), 100)


nbins = 400
frequency = np.linspace(1, 300, nbins)

baseline = wavelet_decomp(lfp[0:2500], lowfreq=1, highfreq=400, nbins=nbins)

# frequency = np.asarray([round(_) for _ in frequency])
# wavedecomp = np.zeros((100, 2500))
timepoints = ripples[[1, 30, 45, 3600, 76], 0]
timepoints = sess.swa.time[5:10]
for i, rpl in enumerate(timepoints):
    start = int(rpl * 1250) - 1250
    end = int(rpl * 1250) + 1250
    signal = lfp[start:end]
    wavedecomp_rpl = wavelet_decomp(signal, lowfreq=1, highfreq=300, nbins=nbins)
    # wavedecomp = wavedecomp + wavedecomp_rpl
    plt.subplot(3, 5, i + 1)
    # bar.update(i)
    # wavedecomp = (wavedecomp_rpl - np.mean(baseline)) / np.std(baseline)
    wavedecomp = wavedecomp_rpl
    plt.pcolormesh(np.linspace(-1, 1, 2500), frequency, wavedecomp, cmap="hot")

    plt.subplot(3, 5, i + 5 + 1)
    plt.plot(np.linspace(-1, 1, 2500), filter_sig.filter_delta(signal), "#aa9d9d")
    plt.subplot(3, 5, i + 10 + 1)
    plt.plot(np.linspace(-1, 1, 2500), stats.zscore(signal), "k")

# plt.colorbar()
# plt.yscale("log")
# plt.yticks(np.arange(0, 100, 10), frequency[np.arange(0, 100, 10)])

# lfpripple = np.asarray(lfpripple)

# wavedecomp1 = wavelet_decomp(lfpripple, lowfreq=1, highfreq=6, nbins=10)
# wavedecomp2 = wavelet_decomp(lfpripple, lowfreq=150, highfreq=250, nbins=20)
# specgrm.append(np.abs(wavedecomp))
# plt.subplot(5, 2, i)
