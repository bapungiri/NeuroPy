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
import scipy.signal as sg

cmap = matplotlib.cm.get_cmap("hot_r")


from callfunc import processData

#%% Subjects
basePath = [
    "/data/Clustering/SleepDeprivation/RatJ/Day1/",
    "/data/Clustering/SleepDeprivation/RatK/Day1/",
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


#%% Gamma across SD
# region
group = []
plt.clf()
fig = plt.figure(1, figsize=(10, 15))
gs = gridspec.GridSpec(1, 3, figure=fig)
fig.subplots_adjust(hspace=0.3)
for sub, sess in enumerate(sessions[:3]):

    sess.trange = np.array([])
    lfp = sess.spindle.best_chan_lfp()[0]
    eegSrate = sess.recinfo.lfpSrate
    post = sess.epochs.post
    sd_period = [post[0], post[0] + 5 * 3600]
    t = np.linspace(0, len(lfp) / eegSrate, len(lfp))

    lfpsd = lfp[(t > sd_period[0]) & (t < sd_period[1])]
    tsd = np.linspace(sd_period[0], sd_period[1], len(lfpsd))
    binsd = np.linspace(sd_period[0], sd_period[1], 6)

    specgram = signal_process.spectrogramBands(lfpsd)

    gamma = stats.zscore(specgram.gamma)
    first_hour_gamma = gamma[(specgram.time > 0) & (specgram.time < 3600)]
    last_hour_gamma = gamma[(specgram.time > 4 * 3600) & (specgram.time < 5 * 3600)]

    hist_first, _ = np.histogram(first_hour_gamma, bins=np.arange(-3, 3, 0.1))
    hist_last, edge = np.histogram(last_hour_gamma, bins=np.arange(-3, 3, 0.1))

    ax = fig.add_subplot(gs[sub])
    ax.plot(edge[:-1], hist_first, "k", label="1st hour")
    ax.plot(edge[:-1], hist_last, "r", label="5th hour")
    ax.set_yscale("log")
    ax.set_xlabel("zscored values")
    ax.set_ylabel("Counts")
    ax.legend()

# endregion

#%% PSD during sleep deprivation from first to last hour to observe gamma reduction
# region
plt.clf()
fig = plt.figure(1, figsize=(10, 15))
gs = gridspec.GridSpec(1, 3, figure=fig)
fig.subplots_adjust(hspace=0.3)

for sub, sess in enumerate(sessions):
    sess.trange = np.array([])
    eegSrate = sess.recinfo.lfpSrate
    channel = sess.theta.bestchan
    post = sess.epochs.post
    eeg = sess.utils.geteeg(chans=channel, timeRange=[post[0], post[0] + 5 * 3600])
    nfrms_hour = eegSrate * 3600
    lfp1st = eeg[:nfrms_hour]
    lfp5th = eeg[-nfrms_hour:]

    psd = lambda sig: sg.welch(
        sig, fs=eegSrate, nperseg=10 * eegSrate, noverlap=5 * eegSrate
    )
    multitaper = lambda sig: signal_process.mtspect(
        sig, fs=eegSrate, nperseg=10 * eegSrate, noverlap=5 * eegSrate
    )

    _, psd1st = multitaper(lfp1st)
    f, psd5th = multitaper(lfp5th)

    ax = fig.add_subplot(gs[sub])
    ax.loglog(f, psd1st, "k", label="ZT1")
    ax.loglog(f, psd5th, "r", label="ZT5")
    ax.set_xlim([1, 120])
    ax.set_ylim([10, 10e5])
    ax.set_title(sess.sessinfo.session.sessionName)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (A.U.)")
    ax.legend()


# endregion

