#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.signal as sg
import matplotlib.gridspec as gridspec
import seaborn as sns
import matplotlib as mpl
from scipy.ndimage import gaussian_filter
from signal_process import spectrogramBands
from sklearn.preprocessing import MinMaxScaler
from callfunc import processData
import signal_process
from scipy import fftpack

#%% Subjects
basePath = [
    "/data/Clustering/SleepDeprivation/RatJ/Day1/",
    "/data/Clustering/SleepDeprivation/RatK/Day1/",
    "/data/Clustering/SleepDeprivation/RatN/Day1/",
    "/data/Clustering/SleepDeprivation/RatJ/Day2/",
    "/data/Clustering/SleepDeprivation/RatK/Day2/",
    "/data/Clustering/SleepDeprivation/RatN/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day4/"
]


sessions = [processData(_) for _ in basePath]

#%% functions
def scale(x):

    x = x - np.min(x)
    x = x / np.max(x)

    return x


def getPxx(lfp):
    window = 5 * 1250

    freq, Pxx = sg.welch(
        lfp, fs=1250, nperseg=window, noverlap=window / 6, detrend="linear",
    )
    noise = np.where(
        ((freq > 59) & (freq < 61)) | ((freq > 119) & (freq < 121)) | (freq > 220)
    )[0]
    freq = np.delete(freq, noise)
    Pxx = np.delete(Pxx, noise)

    return Pxx, freq


#%% Powerspectrum compare during REM
# region
for sub, sess in enumerate(sessions):

    sess.trange = np.array([])

    sampfreq = sess.recinfo.lfpSrate
    maze = sess.epochs.maze
    tstart = sess.epochs.post[0]

    lfp, _, _ = sess.spindle.best_chan_lfp()
    t = np.linspace(0, len(lfp) / 1250, len(lfp))
    deadfile = sess.sessinfo.files.filePrefix.with_suffix(".dead")
    if deadfile.is_file():
        with deadfile.open("r") as f:
            noisy = []
            for line in f:
                epc = line.split(" ")
                epc = [float(_) for _ in epc]
                noisy.append(epc)
            noisy = np.asarray(noisy)
            noisy = ((noisy / 1000) * sampfreq).astype(int)

        for noisy_ind in range(noisy.shape[0]):
            st = noisy[noisy_ind, 0]
            en = noisy[noisy_ind, 1]
            numnoisy = en - st
            lfp[st:en] = np.nan

    states = sess.brainstates.states
    rem = states[(states["start"] > tstart) & (states["name"] == "rem")]

    binlfp = lambda x, t1, t2: x[(t > t1) & (t < t2)]
    lfprem = []
    for epoch in rem.itertuples():
        lfprem.extend(binlfp(lfp, epoch.start, epoch.end))
    lfprem = stats.zscore(np.asarray(lfprem))

    lfpmaze = stats.zscore(binlfp(lfp, maze[0], maze[1]))
    # b, a = sg.iirnotch(60, 30, fs=1250)
    # lfprem = sg.filtfilt(b, a, lfprem)

    sess.pxx_rem, sess.f_rem = getPxx(lfprem)
    sess.pxx_maze, sess.f_maze = getPxx(lfpmaze)


# ====== Plotting ==========
plt.clf()
fig = plt.figure(1, figsize=(15, 8))
gs = GridSpec(1, 3, figure=fig)
# fig.subplots_adjust(hspace=0.5)

for sub, sess in enumerate(sessions):

    if sub < 3:
        plt_ind = sub
        alpha = 1
        shift = 0
    else:
        plt_ind = sub - 3
        alpha = 0.4
        color = "k"
        shift = 2

    subname = sess.sessinfo.session.name

    todB = lambda power: 10 * np.log10(power)

    ax = fig.add_subplot(gs[0, plt_ind])
    plt.plot(sess.f_rem, todB(sess.pxx_rem) - shift, color="#ef253c", alpha=alpha)
    plt.plot(sess.f_maze, todB(sess.pxx_maze) - shift + 6, color=color, alpha=alpha)
    plt.xscale("log")
    # plt.yscale("log")
    ax.set_xlim([4, 220])
    # ax.set_xlim([4, 220])
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("Power (dB)")
    ax.set_title(subname)


ax.legend(["REM-SD", "MAZE-SD", "REM-NSD", "MAZE-NSD"])
fig.suptitle(
    f"Power spectrum REM and MAZE epochs between SD and NSD session. Only REM periods in POST are compared. \n Note: curves have been artificially shifted for clarity"
)
# endregion

#%% Bicoherence plots of REM sleep
# region
plt.clf()
fig = plt.figure(1, figsize=(10, 15))
gs = gridspec.GridSpec(2, 3, figure=fig)
fig.subplots_adjust(hspace=0.3)
for sub, sess in enumerate(sessions):

    sess.trange = np.array([])

    sampfreq = sess.recinfo.lfpSrate
    maze = sess.epochs.maze
    tstart = sess.epochs.post[0]

    lfp, _, _ = sess.spindle.best_chan_lfp()
    t = np.linspace(0, len(lfp) / 1250, len(lfp))
    deadfile = sess.sessinfo.files.filePrefix.with_suffix(".dead")
    if deadfile.is_file():
        with deadfile.open("r") as f:
            noisy = []
            for line in f:
                epc = line.split(" ")
                epc = [float(_) for _ in epc]
                noisy.append(epc)
            noisy = np.asarray(noisy)
            noisy = ((noisy / 1000) * sampfreq).astype(int)

        for noisy_ind in range(noisy.shape[0]):
            st = noisy[noisy_ind, 0]
            en = noisy[noisy_ind, 1]
            numnoisy = en - st
            lfp[st:en] = np.nan

    states = sess.brainstates.states
    rem = states[(states["start"] > tstart) & (states["name"] == "rem")]

    binlfp = lambda x, t1, t2: x[(t > t1) & (t < t2)]
    lfprem = []
    for epoch in rem.itertuples():
        lfprem.extend(binlfp(lfp, epoch.start, epoch.end))
    lfprem = stats.zscore(np.asarray(lfprem))

    # strong_theta = strong_theta - np.mean(strong_theta)
    lfprem = sg.detrend(lfprem, type="linear")
    bicoh, bicoh_freq = signal_process.bicoherence(
        lfprem, window=4 * 1250, overlap=2 * 1250
    )

    ax = fig.add_subplot(gs[sub])
    ax.pcolorfast(bicoh_freq, bicoh_freq, bicoh, cmap="YlGn", vmax=0.2)
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Frequency (Hz)")
    # plt.pcolormesh(bispec_freq, bispec_freq, bispec, vmin=0, vmax=0.1, cmap="YlGn")
    ax.set_ylim([2, 75])

    # ax = fig.add_subplot(gs[sub + 2])
    # f, t, sxx = sg.spectrogram(strong_theta, nperseg=1250, noverlap=625, fs=1250)
    # ax.pcolorfast(t, f, sxx, cmap="YlGn", vmax=0.05)
    # ax.set_ylabel("Frequency (Hz)")
    # ax.set_xlabel("Time (s)")
    # # plt.pcolormesh(bispec_freq, bispec_freq, bispec, vmin=0, vmax=0.1, cmap="YlGn")
    # ax.set_ylim([1, 75])

fig.suptitle("fourier and bicoherence analysis of strong theta during MAZE")
# endregion
