# %%

import random
import warnings

import ipywidgets as widgets
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as sg
import scipy.stats as stats
import seaborn as sns
from matplotlib.widgets import Button, RadioButtons, Slider
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from callfunc import processData
from mathutil import threshPeriods
import signal_process


warnings.simplefilter(action="default")

#%% ====== functions needed for some computation ============


def getspkCorr(spikes, period, binsize=0.1):
    bins = np.arange(period[0], period[1], binsize)
    spk_cnts = np.asarray([np.histogram(cell, bins=bins)[0] for cell in spikes])
    corr = np.corrcoef(spk_cnts)
    np.fill_diagonal(corr, 0)

    return corr


def calculateISI(mua, period, bins):
    mua = mua[np.where((mua > period[0]) & (mua < period[1]))]
    isi = np.diff(mua)
    isihist, _ = np.histogram(isi, bins=bins)

    return isihist


def stability(spikes, period):
    meanfr = np.asarray([len(cell) / np.diff(period) for cell in spikes])
    windows = np.linspace(period[0], period[1], 6)
    meanfr_window = np.asarray(
        [np.histogram(cell, bins=windows)[0] / 3600 for cell in spikes]
    )
    print(meanfr_window.shape)
    fr_fraction = meanfr_window / meanfr

    fr_stable = np.where(fr_fraction > 0.10, 1, 0)
    cells_stable = np.where(np.sum(fr_stable, axis=1) == 5)[0]
    return cells_stable


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


#%% Pairwise correlation change during SD
# region
plt.clf()
fig = plt.figure(1, figsize=(10, 15))
gs = gridspec.GridSpec(1, 2, figure=fig)
fig.subplots_adjust(hspace=0.3)
fig.suptitle("Cross-coherence first hour vs last hour of SD furthest channels")
for sub, sess in enumerate(sessions):
    sess.trange = np.array([])
    post = sess.epochs.post
    eegSrate = sess.recinfo.lfpSrate
    spikes = sess.spikes.times
    sd_period = [post[0], post[0] + 5 * 3600]
    firsthr_time = [post[0], post[0] + 3600]
    fifthhr_time = [post[0] + 4 * 3600, post[0] + 5 * 3600]

    corr_1h = getspkCorr(spikes, firsthr_time)
    corr_5h = getspkCorr(spikes, fifthhr_time)

    subname = sess.sessinfo.session.sessionName
    ax = fig.add_subplot(gs[0])
    ax.imshow(corr_1h, aspect="auto", vmax=0.5)
    ax.set_ylabel("Coherence")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_title(subname)

    ax = fig.add_subplot(gs[1])
    ax.imshow(corr_5h, aspect="auto", vmax=0.5)
    ax.set_ylabel("Coherence")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_title(subname)

# endregion


#%% Change in interspike interval during Sleep Deprivaton
# region
plt.clf()
fig = plt.figure(1, figsize=(10, 15))
gs = gridspec.GridSpec(1, 3, figure=fig)
fig.subplots_adjust(hspace=0.3)
fig.suptitle("Change in interspike interval during Sleep Deprivaton")
for sub, sess in enumerate(sessions):
    sess.trange = np.array([])
    post = sess.epochs.post
    eegSrate = sess.recinfo.lfpSrate
    spikes = sess.spikes.times
    sd_period = [post[0], post[0] + 5 * 3600]
    firsthr_time = [post[0], post[0] + 3600]
    fifthhr_time = [post[0] + 4 * 3600, post[0] + 5 * 3600]
    spkinfo = sess.spikes.info
    reqcells_id = np.where(spkinfo["q"] < 4)[0]
    spikes = [spikes[cell] for cell in reqcells_id]

    mua = np.concatenate(spikes)

    bins = np.arange(0, 0.4, 0.001)
    isi_1h = calculateISI(mua, firsthr_time, bins=bins)
    isi_5h = calculateISI(mua, fifthhr_time, bins=bins)

    subname = sess.sessinfo.session.sessionName
    ax = fig.add_subplot(gs[sub])
    ax.plot(bins[:-1], isi_1h, label="1st")
    ax.plot(bins[:-1], isi_5h, label="5th")
    ax.set_yscale("log")
    # ax.set_xscale("log")
    ax.set_ylabel("Counts")
    ax.set_xlabel("Interspike interval (s)")
    ax.set_title(subname)
    ax.legend()


# endregion

#%% correlation of cells which are stable during SD
# region

plt.clf()
fig = plt.figure(1, figsize=(10, 15))
gs = gridspec.GridSpec(1, 1, figure=fig)
fig.subplots_adjust(hspace=0.3)
fig.suptitle("Change in interspike interval during Sleep Deprivaton")
for sub, sess in enumerate(sessions):
    sess.trange = np.array([])
    post = sess.epochs.post
    eegSrate = sess.recinfo.lfpSrate
    spikes = sess.spikes.times
    sd_period = [post[0], post[0] + 5 * 3600]
    firsthr_time = [post[0], post[0] + 3600]
    fifthhr_time = [post[0] + 4 * 3600, post[0] + 5 * 3600]
    spkinfo = sess.spikes.info
    reqcells_id = np.where(spkinfo["q"] < 4)[0]
    spikes = [spikes[cell] for cell in reqcells_id]

    stable_cells = stability(spikes, sd_period)
    spikes = [spikes[cell] for cell in stable_cells]

    corr_1h = getspkCorr(spikes, firsthr_time)
    corr_5h = getspkCorr(spikes, fifthhr_time)

    meancorr_1h = np.mean(corr_1h[np.tril_indices_from(corr_1h,)])
    meancorr_5h = np.mean(corr_5h[np.tril_indices_from(corr_5h,)])

    subname = sess.sessinfo.session.sessionName
    ax = fig.add_subplot(gs[0])
    ax.plot([1, 2], [meancorr_1h, meancorr_5h])
    ax.set_ylabel("Coherence")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_title(subname)
# endregion


#%% participation rate during ripples
# region

plt.clf()
fig = plt.figure(1, figsize=(10, 15))
gs = gridspec.GridSpec(1, 1, figure=fig)
fig.subplots_adjust(hspace=0.3)
fig.suptitle("Change in interspike interval during Sleep Deprivaton")
for sub, sess in enumerate(sessions):
    sess.trange = np.array([])
    post = sess.epochs.post
    eegSrate = sess.recinfo.lfpSrate
    spikes = sess.spikes.times
    sd_period = [post[0], post[0] + 5 * 3600]
    firsthr_time = [post[0], post[0] + 3600]
    fifthhr_time = [post[0] + 4 * 3600, post[0] + 5 * 3600]
    spkinfo = sess.spikes.info
    reqcells_id = np.where(spkinfo["q"] < 4)[0]
    spikes = [spikes[cell] for cell in reqcells_id]

    stable_cells = stability(spikes, sd_period)
    spikes = [spikes[cell] for cell in stable_cells]

    corr_1h = getspkCorr(spikes, firsthr_time)
    corr_5h = getspkCorr(spikes, fifthhr_time)

    meancorr_1h = np.mean(corr_1h[np.tril_indices_from(corr_1h,)])
    meancorr_5h = np.mean(corr_5h[np.tril_indices_from(corr_5h,)])

    subname = sess.sessinfo.session.sessionName
    ax = fig.add_subplot(gs[0])
    ax.plot([1, 2], [meancorr_1h, meancorr_5h])
    ax.set_ylabel("Coherence")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_title(subname)
# endregion
