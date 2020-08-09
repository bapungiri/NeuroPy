# %%

import random
import warnings

# import ipywidgets as widgets
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
from ccg import correlograms
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
        [
            np.histogram(cell, bins=windows)[0] / np.mean(np.diff(windows))
            for cell in spikes
        ]
    )
    print(meanfr_window.shape)
    fr_fraction = meanfr_window / meanfr

    fr_stable = np.where(fr_fraction > 0.3, 1, 0)
    cells_stable = np.where(np.sum(fr_stable, axis=1) == len(windows) - 1)[0]
    return cells_stable


#%% Subjects to choose from
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


#%% Pairwise correlation change during SD
# region
plt.clf()
fig = plt.figure(1, figsize=(10, 15))
gs = gridspec.GridSpec(3, 2, figure=fig)
fig.subplots_adjust(hspace=0.3)
fig.suptitle("Cross-coherence first hour vs last hour of SD furthest channels")
for sub, sess in enumerate(sessions[:3]):
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
    ax = fig.add_subplot(gs[sub, 0])
    ax.imshow(corr_1h, aspect="auto", vmax=0.1, vmin=-0.1)
    ax.set_ylabel("Coherence")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_title(subname)

    ax = fig.add_subplot(gs[sub, 1])
    ax.imshow(corr_5h, aspect="auto", vmax=0.1, vmin=-0.1)
    ax.set_ylabel("Coherence")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_title(subname)

# endregion

#%% Mean pairwise correlation across Sleep Deprivation
# region

plt.clf()
fig = plt.figure(1, figsize=(15, 5))
gs = gridspec.GridSpec(1, 3, figure=fig)
fig.subplots_adjust(hspace=0.3)
fig.suptitle("Mean pairwise correlation")
for sub, sess in enumerate(sessions[:3]):
    sess.trange = np.array([])
    post = sess.epochs.post
    eegSrate = sess.recinfo.lfpSrate
    spikes = sess.spikes.times
    spkinfo = sess.spikes.info
    nCells = len(spkinfo)

    good_pyr = np.where(spkinfo.q < 4)[0]
    mua = np.where(spkinfo.q == 6)[0]
    intneur = np.where(spkinfo.q == 8)[0]

    indx_pyr = np.ix_(good_pyr, good_pyr)
    indx_mua = np.ix_(mua, mua)
    indx_intr = np.ix_(intneur, intneur)

    sd_period = [post[0], post[0] + 5 * 3600]
    sd_period_bin = np.arange(post[0], post[0] + 5 * 3600, 300)

    corr = [
        getspkCorr(spikes, [sd_period_bin[i], sd_period_bin[i + 1]])
        for i in range(len(sd_period_bin) - 1)
    ]

    lower_tr = np.tril_indices(nCells, k=-1)
    lower_pyr = np.tril_indices(len(good_pyr), k=-1)
    lower_mua = np.tril_indices(len(mua), k=-1)
    lower_intr = np.tril_indices(len(intneur), k=-1)

    mean_corr = np.asarray([np.nanmean(mat[lower_tr]) for mat in corr])
    mean_corr_pyr = np.asarray([np.nanmean(mat[indx_pyr][lower_pyr]) for mat in corr])
    mean_corr_mua = np.asarray([np.nanmean(mat[indx_mua][lower_mua]) for mat in corr])
    mean_corr_intr = np.asarray(
        [np.nanmean(mat[indx_intr][lower_intr]) for mat in corr]
    )

    subname = sess.sessinfo.session.sessionName
    t = np.linspace(0, 5, len(mean_corr) + 1)
    ax = fig.add_subplot(gs[sub])
    ax.plot(t[:-1], mean_corr, "k")
    ax.plot(t[:-1], mean_corr_pyr, "r")
    ax.plot(t[:-1], mean_corr_mua, "gray")
    ax.plot(t[:-1], mean_corr_intr, "g")
    ax.set_ylabel("Mean correlation")
    ax.set_xlabel("Time (h)")
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
fig = plt.figure(1, figsize=(10, 5))
gs = gridspec.GridSpec(1, 3, figure=fig)
fig.subplots_adjust(hspace=0.3)
# fig.suptitle("Change in interspike interval during Sleep Deprivaton")
corr_all_1h, corr_all_5h = [], []
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

    corr_1h = corr_1h[np.tril_indices_from(corr_1h, k=-1)]
    corr_5h = corr_5h[np.tril_indices_from(corr_5h, k=-1)]

    corr_all_1h.append(corr_1h)
    corr_all_5h.append(corr_5h)

    meancorr_1h = np.mean(corr_1h)
    meancorr_5h = np.mean(corr_5h)

    subname = sess.sessinfo.session.sessionName
    axmeancorr = fig.add_subplot(gs[2])
    axmeancorr.plot([1, 2], [meancorr_1h, meancorr_5h], "o-", color="gray")


axmeancorr.set_ylabel("Mean correlation")
axmeancorr.set_xlabel("Bins")
axmeancorr.set_title("Mean correlation across sleep deprivation")
axmeancorr.set_xlim([0, 3])
axmeancorr.set_ylim([0.025, 0.1])
axmeancorr.set_xticks([1, 2])
axmeancorr.set_xticklabels(["ZT1", "ZT5"])


axdistcorr = fig.add_subplot(gs[0])
hist_corr_1h, edges = np.histogram(
    np.concatenate(corr_all_1h[:3]), bins=np.linspace(-0.1, 0.3, 30)
)
hist_corr_5h, edges = np.histogram(
    np.concatenate(corr_all_5h[:3]), bins=np.linspace(-0.1, 0.3, 30)
)
hist_corr_1h_nsd, edges = np.histogram(
    np.concatenate(corr_all_1h[3:]), bins=np.linspace(-0.1, 0.3, 30)
)
hist_corr_5h_nsd, edges = np.histogram(
    np.concatenate(corr_all_5h[3:]), bins=np.linspace(-0.1, 0.3, 30)
)
# axdistcorr.plot(edges[:-1], hist_corr_1h)
# axdistcorr.plot(edges[:-1], hist_corr_5h)
axdistcorr.plot(edges[:-1], np.cumsum(hist_corr_1h) / np.sum(hist_corr_1h), "r")
axdistcorr.plot(edges[:-1], np.cumsum(hist_corr_5h) / np.sum(hist_corr_5h), "g")
axdistcorr.plot(edges[:-1], np.cumsum(hist_corr_1h_nsd) / np.sum(hist_corr_1h_nsd), "k")
axdistcorr.plot(
    edges[:-1], np.cumsum(hist_corr_5h_nsd) / np.sum(hist_corr_5h_nsd), "gray"
)
axdistcorr.set_xlabel("correlation")
axdistcorr.set_ylabel("cdf")

# axcorr = fig.add_subplot(gs[1])
# axcorr.plot(corr_all_1h, corr_all_5h, "k.")


# endregion


#%% Number of spike during ripples over the course of sleep deprivation
# region

plt.clf()
fig = plt.figure(1, figsize=(10, 15))
gs = gridspec.GridSpec(3, 1, figure=fig)
fig.subplots_adjust(hspace=0.3)
fig.suptitle("Change in interspike interval during Sleep Deprivaton")
for sub, sess in enumerate(sessions):
    sess.trange = np.array([])
    post = sess.epochs.post
    eegSrate = sess.recinfo.lfpSrate
    spikes = sess.spikes.times
    sd_period = [post[0], post[0] + 5 * 3600]
    ripples = sess.ripple.time
    ripples_sd = ripples[
        (ripples[:, 0] > sd_period[0]) & (ripples[:, 0] < sd_period[1])
    ].ravel()

    spkcnt = np.asarray([np.histogram(cell, ripples_sd)[0] for cell in spikes])[:, ::2]

    ax = fig.add_subplot(gs[sub])
    ax.plot(np.sum(spkcnt, axis=0))
    ax.set_ylabel("Spike counts")
    ax.set_xlabel("time")
    # ax.set_title(subname)


# endregion

#%% CCG temporal shift from first hour of SD to 5th hour of SD
# region

plt.clf()
fig = plt.figure(1, figsize=(10, 15))
gs = gridspec.GridSpec(3, 1, figure=fig)
fig.subplots_adjust(hspace=0.3)
fig.suptitle("Change in interspike interval during Sleep Deprivaton")
for sub, sess in enumerate(sessions):
    sess.trange = np.array([])
    post = sess.epochs.post
    eegSrate = sess.recinfo.lfpSrate
    spikes = sess.spikes.times
    spkid = np.arange(1, len(spikes) + 1).astype(int)

    sd_period = [post[0], post[0] + 1 * 3600]
    spikes_sd = [cell[(cell > sd_period[0]) & (cell < sd_period[1])] for cell in spikes]
    clu_map = [[spkid[i]] * len(spikes_sd[i]) for i in range(len(spikes))]
    spikes_sd = np.concatenate(spikes_sd)
    clu_map = np.concatenate(clu_map).astype(int)
    a = np.unique(clu_map)
    sort_ind = np.argsort(spikes_sd)
    spikes_sd = spikes_sd[sort_ind]
    clu_map = clu_map[sort_ind]
    ccgmap = correlograms(
        spikes_sd, clu_map, sample_rate=1250, bin_size=0.01, window_size=0.5
    )
    ccg_lag = ccgmap[:, :, 0:25].sum(axis=2)
    ccg_lead = ccgmap[:, :, 26:51].sum(axis=2)
    diff_ccg_1st = ccg_lead - ccg_lag

    sd_period = [post[0] + 4 * 3600, post[0] + 5 * 3600]
    spikes_sd = [cell[(cell > sd_period[0]) & (cell < sd_period[1])] for cell in spikes]
    clu_map = [[spkid[i]] * len(spikes_sd[i]) for i in range(len(spikes))]
    spikes_sd = np.concatenate(spikes_sd)
    clu_map = np.concatenate(clu_map).astype(int)
    b = np.unique(clu_map)
    sort_ind = np.argsort(spikes_sd)
    spikes_sd = spikes_sd[sort_ind]
    clu_map = clu_map[sort_ind]
    ccgmap = correlograms(
        spikes_sd, clu_map, sample_rate=1250, bin_size=0.01, window_size=0.5
    )
    ccg_lag = ccgmap[:, :, 0:25].sum(axis=2)
    ccg_lead = ccgmap[:, :, 26:51].sum(axis=2)
    diff_ccg_5th = ccg_lead - ccg_lag

    if len(a) > len(b):
        indx = np.argwhere(np.isin(a, b)).squeeze()
        indx = np.ix_(indx, indx)
        diff_ccg_1st = diff_ccg_1st[indx]
    if len(b) > len(a):
        indx = np.argwhere(np.isin(b, a)).squeeze()

        indx = np.ix_(indx, indx)
        diff_ccg_5th = diff_ccg_5th[indx]

    low_indices = np.tril_indices(np.min([len(a), len(b)]), k=-1)
    first_hour = diff_ccg_1st[low_indices]
    fifth_hour = diff_ccg_5th[low_indices]

    linfit = np.polyfit(first_hour, fifth_hour, 1)
    print(linfit)

    ax = fig.add_subplot(gs[sub])
    ax.plot(first_hour, fifth_hour, ".")
    ax.set_ylabel("Spike counts")
    ax.set_xlabel("time")
    # ax.set_title(subname)


# endregion

#%% CCG temporal shift NREM recovery vs NREM control session
# region

plt.clf()
fig = plt.figure(1, figsize=(10, 15))
gs = gridspec.GridSpec(3, 1, figure=fig)
fig.subplots_adjust(hspace=0.3)
fig.suptitle("Change in interspike interval during Sleep Deprivaton")
for sub, sess in enumerate(sessions):
    sess.trange = np.array([])
    maze = sess.epochs.maze
    post = sess.epochs.post
    eegSrate = sess.recinfo.lfpSrate
    spikes = sess.spikes.times
    spkid = np.arange(1, len(spikes) + 1).astype(int)

    sd_period = [post[0], post[0] + 1 * 3600]
    spikes_sd = [cell[(cell > sd_period[0]) & (cell < sd_period[1])] for cell in spikes]
    clu_map = [[spkid[i]] * len(spikes_sd[i]) for i in range(len(spikes))]
    spikes_sd = np.concatenate(spikes_sd)
    clu_map = np.concatenate(clu_map).astype(int)
    a = np.unique(clu_map)
    sort_ind = np.argsort(spikes_sd)
    spikes_sd = spikes_sd[sort_ind]
    clu_map = clu_map[sort_ind]
    ccgmap = correlograms(
        spikes_sd, clu_map, sample_rate=1250, bin_size=0.01, window_size=0.5
    )
    ccg_lag = ccgmap[:, :, 0:25].sum(axis=2)
    ccg_lead = ccgmap[:, :, 26:51].sum(axis=2)
    diff_ccg_1st = ccg_lead - ccg_lag

    sd_period = [post[0] + 4 * 3600, post[0] + 5 * 3600]
    spikes_sd = [cell[(cell > sd_period[0]) & (cell < sd_period[1])] for cell in spikes]
    clu_map = [[spkid[i]] * len(spikes_sd[i]) for i in range(len(spikes))]
    spikes_sd = np.concatenate(spikes_sd)
    clu_map = np.concatenate(clu_map).astype(int)
    b = np.unique(clu_map)
    sort_ind = np.argsort(spikes_sd)
    spikes_sd = spikes_sd[sort_ind]
    clu_map = clu_map[sort_ind]
    ccgmap = correlograms(
        spikes_sd, clu_map, sample_rate=1250, bin_size=0.01, window_size=0.5
    )
    ccg_lag = ccgmap[:, :, 0:25].sum(axis=2)
    ccg_lead = ccgmap[:, :, 26:51].sum(axis=2)
    diff_ccg_5th = ccg_lead - ccg_lag

    if len(a) > len(b):
        indx = np.argwhere(np.isin(a, b)).squeeze()
        indx = np.ix_(indx, indx)
        diff_ccg_1st = diff_ccg_1st[indx]
    if len(b) > len(a):
        indx = np.argwhere(np.isin(b, a)).squeeze()

        indx = np.ix_(indx, indx)
        diff_ccg_5th = diff_ccg_5th[indx]

    low_indices = np.tril_indices(np.min([len(a), len(b)]), k=-1)
    first_hour = diff_ccg_1st[low_indices]
    fifth_hour = diff_ccg_5th[low_indices]

    linfit = np.polyfit(first_hour, fifth_hour, 1)
    print(linfit)

    ax = fig.add_subplot(gs[sub])
    ax.plot(first_hour, fifth_hour, ".")
    ax.set_ylabel("Spike counts")
    ax.set_xlabel("time")
    # ax.set_title(subname)


# endregion

#%% correlation of cells which are stable(MAZE to POST) during SD
# region

plt.clf()
fig = plt.figure(1, figsize=(10, 5))
gs = gridspec.GridSpec(2, 2, figure=fig)
fig.subplots_adjust(hspace=0.3)
# fig.suptitle("Change in interspike interval during Sleep Deprivaton")
corr_all_maze, corr_all_zt_1to5, corr_all_zt_5to10 = [], [], []
for sub, sess in enumerate(sessions):
    sess.trange = np.array([])
    post = sess.epochs.post
    maze = sess.epochs.maze
    eegSrate = sess.recinfo.lfpSrate
    spikes = sess.spikes.times
    period = [maze[0], post[1]]

    zt_1to5 = [post[0], post[0] + 5 * 3600]
    zt_5to10 = [post[0] + 5 * 3600, post[0] + 10 * 3600]

    spkinfo = sess.spikes.info
    reqcells_id = np.where(spkinfo["q"] < 4)[0]
    spikes = [spikes[cell] for cell in reqcells_id]

    stable_cells = stability(spikes, period)
    spikes = [spikes[cell] for cell in stable_cells]

    corr_maze = getspkCorr(spikes, maze)
    corr_5h = getspkCorr(spikes, zt_1to5)
    corr_last = getspkCorr(spikes, zt_5to10)

    corr_maze = corr_maze[np.tril_indices_from(corr_maze, k=-1)]
    corr_5h = corr_5h[np.tril_indices_from(corr_5h, k=-1)]
    corr_last = corr_last[np.tril_indices_from(corr_last, k=-1)]

    corr_all_maze.append(corr_maze)
    corr_all_zt_1to5.append(corr_5h)
    corr_all_zt_5to10.append(corr_last)

    meancorr_1h = np.mean(corr_maze)
    meancorr_5h = np.mean(corr_5h)

    subname = sess.sessinfo.session.sessionName
#     axmeancorr = fig.add_subplot(gs[2])
#     axmeancorr.plot([1, 2], [meancorr_1h, meancorr_5h], "o-", color="gray")


# axmeancorr.set_ylabel("Mean correlation")
# axmeancorr.set_xlabel("Bins")
# axmeancorr.set_title("Mean correlation across sleep deprivation")
# axmeancorr.set_xlim([0, 3])
# axmeancorr.set_ylim([0.025, 0.1])
# axmeancorr.set_xticks([1, 2])
# axmeancorr.set_xticklabels(["ZT1", "ZT5"])


axdistcorr = fig.add_subplot(gs[0, 0])

axdistcorr.plot(
    np.concatenate(corr_all_maze[:3]), np.concatenate(corr_all_zt_1to5[:3]), "."
)
print(
    np.corrcoef(
        np.concatenate(corr_all_maze[:3]), np.concatenate(corr_all_zt_1to5[:3])
    )[0, 1]
)
axdistcorr.set_xlim([-0.2, 0.4])
axdistcorr.set_ylim([-0.2, 0.4])

axdistcorr = fig.add_subplot(gs[0, 1])

axdistcorr.plot(
    np.concatenate(corr_all_maze[:3]), np.concatenate(corr_all_zt_5to10[:3]), "."
)
print(
    np.corrcoef(
        np.concatenate(corr_all_maze[:3]), np.concatenate(corr_all_zt_5to10[:3])
    )[0, 1]
)
axdistcorr.set_xlim([-0.2, 0.4])
axdistcorr.set_ylim([-0.2, 0.4])

# hist_corr_maze_sd, edges = np.histogram(
#     np.concatenate(corr_all_maze[:3]), bins=np.linspace(-0.1, 0.3, 30)
# )
# hist_corr_1to5h_sd, edges = np.histogram(
#     np.concatenate(corr_all_zt_1to5[:3]), bins=np.linspace(-0.1, 0.3, 30)
# )
# hist_corr_5to10_sd, edges = np.histogram(
#     np.concatenate(corr_all_zt_5to10[:3]), bins=np.linspace(-0.1, 0.3, 30)
# )
# hist_corr_maze_nsd, edges = np.histogram(
#     np.concatenate(corr_all_maze[3:]), bins=np.linspace(-0.1, 0.3, 30)
# )
# hist_corr_1to5h_nsd, edges = np.histogram(
#     np.concatenate(corr_all_zt_1to5[3:]), bins=np.linspace(-0.1, 0.3, 30)
# )
# hist_corr_5to10_nsd, edges = np.histogram(
#     np.concatenate(corr_all_zt_5to10[3:]), bins=np.linspace(-0.1, 0.3, 30)
# )
# axdistcorr.plot(edges[:-1], hist_corr_1h)
# axdistcorr.plot(edges[:-1], hist_corr_5h)
# axdistcorr.plot(
#     edges[:-1], np.cumsum(hist_corr_maze_sd) / np.sum(hist_corr_maze_sd), "#455A64"
# )
# axdistcorr.plot(
#     edges[:-1], np.cumsum(hist_corr_1to5h_sd) / np.sum(hist_corr_1to5h_sd), "#FF8A65"
# )
# axdistcorr.plot(
#     edges[:-1], np.cumsum(hist_corr_5to10_sd) / np.sum(hist_corr_5to10_sd), "#009688"
# )

# axdistcorr.set_xlabel("correlation")
# axdistcorr.set_ylabel("cdf")
# axdistcorr.legend(["MAZE", "POST(1-5h)", "POST(5-10h)"])

axdistcorr = fig.add_subplot(gs[1, 0])
axdistcorr.plot(
    np.concatenate(corr_all_maze[3:]), np.concatenate(corr_all_zt_1to5[3:]), "."
)
print(
    np.corrcoef(
        np.concatenate(corr_all_maze[3:]), np.concatenate(corr_all_zt_1to5[3:])
    )[0, 1]
)
axdistcorr.set_xlim([-0.2, 0.5])
axdistcorr.set_ylim([-0.2, 0.5])

axdistcorr = fig.add_subplot(gs[1, 1])

axdistcorr.plot(
    np.concatenate(corr_all_maze[3:]), np.concatenate(corr_all_zt_5to10[3:]), "."
)
print(
    np.corrcoef(
        np.concatenate(corr_all_maze[3:]), np.concatenate(corr_all_zt_5to10[3:])
    )[0, 1]
)
axdistcorr.set_xlim([-0.2, 0.5])
axdistcorr.set_ylim([-0.2, 0.5])
# axdistcorr.plot(
#     edges[:-1], np.cumsum(hist_corr_maze_nsd) / np.sum(hist_corr_maze_nsd), "#455A64"
# )
# axdistcorr.plot(
#     edges[:-1], np.cumsum(hist_corr_1to5h_nsd) / np.sum(hist_corr_1to5h_nsd), "#FF8A65"
# )
# axdistcorr.plot(
#     edges[:-1], np.cumsum(hist_corr_5to10_nsd) / np.sum(hist_corr_5to10_nsd), "#009688"
# )

axdistcorr.set_xlabel("pairwise correlation")
axdistcorr.set_ylabel("cdf")


# axcorr = fig.add_subplot(gs[1])
# axcorr.plot(corr_all_1h, corr_all_5h, "k.")


# endregion

#%% Correlation of corrleation between MAZE and POST
# region
"""Stable units only from MAZE to POST
"""
plt.clf()
fig = plt.figure(1, figsize=(6, 10))
gs = gridspec.GridSpec(3, 1, figure=fig)
fig.subplots_adjust(hspace=0.4)

for sub, sess in enumerate(sessions):

    sess.trange = np.array([])

    sess.spikes.stability.firingRate()
    # corr = sess.replay.correlation()
    # plt.plot(corr)
# endregion

