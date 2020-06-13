import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.signal as sg
from matplotlib.gridspec import GridSpec
import seaborn as sns
import matplotlib as mpl
from scipy.ndimage import gaussian_filter
from signal_process import spectrogramBands
from sklearn.preprocessing import MinMaxScaler
from callfunc import processData
from signal_process import wavelet_decomp

basePath = [
    # "/data/Clustering/SleepDeprivation/RatJ/Day1/",
    # "/data/Clustering/SleepDeprivation/RatK/Day1/",
    # "/data/Clustering/SleepDeprivation/RatN/Day1/",
    "/data/Clustering/SleepDeprivation/RatJ/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day2/",
    # "/data/Clustering/SleepDeprivation/RatN/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day4/"
]


sessions = [processData(_) for _ in basePath]

# during REM sleep
plt.close("all")
# fig.subplots_adjust(hspace=0.5)


def scale(x):

    x = x - np.min(x)
    x = x / np.max(x)

    return x


for sub, sess in enumerate(sessions):

    sess.trange = np.array([])

    sampfreq = sess.recinfo.lfpSrate
    tstart = sess.epochs.post[0]
    tend = sess.epochs.post[0] + 5 * 3600

    lfp, _, _ = sess.spindle.best_chan_lfp()
    t = np.linspace(0, len(lfp) / 1250, len(lfp))
    states = sess.brainstates.states

    if sub < 3:
        plt_ind = sub
        # color = "r"
        # color = colband[sub]
        lnstyle = "solid"
        rem = states[(states["start"] > tstart) & (states["name"] == "rem")]
    else:
        plt_ind = sub - 3
        # color = colband[sub - 3]
        lnstyle = "dashed"
        rem = states[(states["start"] > tstart) & (states["name"] == "rem")]

    binlfp = lambda x, t1, t2: x[(t > t1) & (t < t2)]

    fig = plt.figure(sub + 1, figsize=(10, 15))
    gs = GridSpec(5, 1, figure=fig)

    lfprem = []
    for epoch in rem.itertuples():
        lfprem.extend(binlfp(lfp, epoch.start, epoch.end))

    lfprem = scale(np.asarray(lfprem))
    lfprem_t = np.linspace(0, len(lfprem) / sampfreq, len(lfprem))

    b, a = sg.iirnotch(60, 30, fs=1250)
    lfprem = sg.filtfilt(b, a, lfprem)
    specgram = spectrogramBands(lfprem, window=5 * 1250)

    norm = 1 / specgram.freq

    norm = np.repeat(norm.reshape(-1, 1), specgram.sxx.shape[1], axis=1)

    sxx = specgram.sxx / norm

    # sxx = specgram.sxx / np.max(specgram.sxx)
    sxx = stats.zscore(gaussian_filter(sxx, sigma=0.2))
    # vmax = np.max(sxx) / 60

    ax = fig.add_subplot(gs[0, 0])
    ax.pcolorfast(specgram.time, specgram.freq, sxx, cmap="YlGn", vmax=1)
    ax.set_ylim([0, 150])

    # ax = fig.add_subplot(gs[1, 0])
    # window = 2 * 1250
    # freq, Pxx = sg.welch(lfprem, fs=1250, nperseg=window, noverlap=window / 2)
    # plt.plot(freq, Pxx)
    # plt.xscale("log")
    # plt.yscale("log")
    # ax.set_xlim([2, 150])

    ax = fig.add_subplot(gs[2, 0])
    # for gamma range
    wavefreqs = np.arange(20, 150, 3)

    wavdec = wavelet_decomp(lfprem, freqs=wavefreqs)
    wavcolgin = wavdec.colgin2009()
    wavcolgin = gaussian_filter(wavcolgin, sigma=2)
    ax.pcolorfast(lfprem_t, wavefreqs, wavcolgin, cmap="jet", vmax=5)

    ax = fig.add_subplot(gs[3, 0], sharex=ax)
    # for theta range
    wavdec.freqs = np.arange(4, 16, 1)
    thetacolgin = wavdec.colgin2009()
    ax.pcolorfast(lfprem_t, wavdec.freqs, thetacolgin, cmap="jet")

    # wavtallon = wavdec.tallonBaudry()
    # wav = 10 * np.log10(wav)
    # plt.imshow(wavtallon, aspect="auto", cmap="jet")

    ax = fig.add_subplot(gs[4, 0], sharex=ax)
    plt.plot(lfprem_t, lfprem, "k")


# freq = specgram.freq
# a = 1 / specgram.freq
# b = specgram.sxx[:, 0]
# plt.plot(freq, b / a)
