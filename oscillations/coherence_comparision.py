# %%

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as sg
import scipy.stats as stats
import seaborn as sns
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button, RadioButtons, Slider
from scipy.ndimage import gaussian_filter, gaussian_filter1d
import ipywidgets as widgets
import random

from callfunc import processData
from signal_process import filter_sig, hilbertfast, wavelet_decomp, spectrogramBands
from mathutil import threshPeriods
import warnings

warnings.simplefilter(action="default")

#%% ====== functions needed for some computation ============


def doWavelet(lfp, freqs, ncycles=3):
    wavdec = wavelet_decomp(lfp, freqs=freqs)
    # wav = wavdec.cohen(ncycles=ncycles)
    wav = wavdec.colgin2009()
    wav = stats.zscore(wav)
    wav = gaussian_filter(wav, sigma=4)
    return wav


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


#%% Subjects to choose from
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

#%% Cross-coherence first hour vs last hour of SD furthest channels
# region
plt.clf()
fig = plt.figure(1, figsize=(10, 15))
gs = gridspec.GridSpec(3, 1, figure=fig)
fig.subplots_adjust(hspace=0.3)
fig.suptitle("Cross-coherence first hour vs last hour of SD furthest channels")
for sub, sess in enumerate(sessions):
    sess.trange = np.array([])
    post = sess.epochs.post
    sampfreq = sess.recinfo.lfpSrate
    channels = sess.recinfo.channels
    badchans = sess.recinfo.badchans
    goodchans = np.setdiff1d(channels, badchans, assume_unique=True)
    firstchan = goodchans[0]
    lastchan = goodchans[-1]

    firsthr_time = [post[0], post[0] + 3600]
    fifthhr_time = [post[0] + 4 * 3600, post[0] + 5 * 3600]

    eeg1sthr = sess.utils.geteeg([firstchan, lastchan], firsthr_time)
    eeg5thhr = sess.utils.geteeg([firstchan, lastchan], fifthhr_time)

    freq1, cxx1st = sg.coherence(
        eeg1sthr[0, :], eeg1sthr[1, :], fs=sampfreq, nperseg=10 * sampfreq
    )
    freq2, cxx5th = sg.coherence(
        eeg5thhr[0, :], eeg5thhr[1, :], fs=sampfreq, nperseg=10 * sampfreq
    )

    subname = sess.sessinfo.session.sessionName
    ax = fig.add_subplot(gs[sub])
    ax.plot(freq1, cxx1st, label="1st", color="#545a6d")
    ax.plot(freq2, cxx5th, label="5th", color="#ff5f5c")
    ax.legend()
    ax.set_xscale("log")
    ax.set_ylabel("Coherence")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_title(subname)


# endregion


# %% instantenous firing rate - lfp coherence
# region
plt.clf()
fig = plt.figure(1, figsize=(10, 15))
gs = gridspec.GridSpec(3, 1, figure=fig)
fig.subplots_adjust(hspace=0.3)
fig.suptitle("Instantenous firing rate - lfp coherence during SD")
for sub, sess in enumerate(sessions):
    sess.trange = np.array([])
    post = sess.epochs.post
    sampfreq = sess.recinfo.lfpSrate
    channels = sess.recinfo.channels
    badchans = sess.recinfo.badchans
    goodchans = np.setdiff1d(channels, badchans)
    firstchan = goodchans[0]
    lastchan = goodchans[-1]
    instfiring = sess.localsleep.instfiring
    instfr_period = sess.localsleep.period
    t_instfiring = np.linspace(instfr_period[0], instfr_period[1], len(instfiring))

    firsthr_time = [post[0], post[0] + 3600]
    fifthhr_time = [post[0] + 4 * 3600, post[0] + 5 * 3600]

    instfr_1sthr = instfiring[
        (t_instfiring > firsthr_time[0]) & (t_instfiring < firsthr_time[1])
    ]
    instfr_5thhr = instfiring[
        (t_instfiring > fifthhr_time[0]) & (t_instfiring < fifthhr_time[1])
    ]

    eeg1sthr = sess.recinfo.geteeg([firstchan, lastchan], firsthr_time)[0]
    eeg5thhr = sess.recinfo.geteeg([firstchan, lastchan], fifthhr_time)[0]

    eeg1sthr = sg.resample(eeg1sthr, num=3600 * 1000)
    eeg5thhr = sg.resample(eeg5thhr, num=3600 * 1000)

    freq1, cxx1st = sg.coherence(eeg1sthr, instfr_1sthr, fs=1000, nperseg=10 * sampfreq)
    freq2, cxx5th = sg.coherence(eeg5thhr, instfr_5thhr, fs=1000, nperseg=10 * sampfreq)

    subname = sess.sessinfo.session.sessionName
    ax = fig.add_subplot(gs[sub])
    ax.plot(freq1, cxx1st, label="1st", color="#545a6d")
    ax.plot(freq2, cxx5th, label="5th", color="#ff5f5c")
    ax.legend()
    ax.set_xlim([2, 300])
    ax.set_xscale("log")
    ax.set_ylabel("Coherence")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_title(subname)
# endregion


# %%
