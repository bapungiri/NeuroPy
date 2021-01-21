# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as sg
import scipy.stats as stats
import seaborn as sns
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from ccg import correlograms
import signal_process
from plotUtil import Fig
import subjects

# warnings.simplefilter(action="default")

#%% ====== functions needed for some computation ============
# region
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


# endregion

#%% Example raster plot during theta cycle and ripple events
# region
"""
good periods for raster plots for some sessions
    RatNDay2: 
        [12672, 12679] (both theta and ripples)
        [13348, 13360] (both theta and ripples)
        [12426, 12436] (theta only)

"""
figure = Fig()
fig, gs = figure.draw(grid=(5, 1), size=(6, 5))
for sess in sessions:
    maze = sess.epochs.maze
    # period = maze
    period = [12674, 12678.5]
    ripples = sess.ripple.events
    ripples = ripples[(ripples.start > period[0]) & (ripples.start < period[1])]
    lfpmaze = sess.recinfo.geteeg(chans=sess.theta.bestchan, timeRange=period)
    ripple_lfp = sess.recinfo.geteeg(chans=sess.ripple.bestchans[4], timeRange=period)
    ripple_lfp = signal_process.filter_sig.ripple(ripple_lfp)
    lfp_t = np.linspace(period[0], period[1], len(lfpmaze)) - period[0]

    # ----- lfp plot --------------
    ax = plt.subplot(gs[0])
    ax.plot(lfp_t, lfpmaze, "k")
    ax.plot(
        ripples.peaktime - period[0], 4200 * np.ones(len(ripples)), "*", color="#f4835d"
    )
    ax.plot([0, 0], [0, 2500], "k", lw=3)  # lfp scale bar 2.5 mV
    ax.axis("off")
    ax.set_title("CA1 local field potential", loc="left")
    ax.annotate(
        "Sharp-wave ripple events",
        xy=(0, 0.9),
        xycoords="axes fraction",
        color="#f4835d",
    )
    ax.set_xlim(left=0)
    ax.set_ylim([-5000, 5000])

    # ------ ripple band plot -----------
    # axripple = plt.subplot(gs[1], sharex=ax)
    # axripple.plot(lfp_t, stats.zscore(ripple_lfp), "gray", lw=0.8)
    # axripple.set_ylim([-12, 12])
    # axripple.axis("off")
    # axripple.set_title("Ripple band (150-250 Hz)", loc="left")

    # ----- raster plot -----------
    axraster = plt.subplot(gs[1:], sharex=ax)
    sess.spikes.plot_raster(
        # spikes=sess.spikes.pyr,
        period=period,
        ax=axraster,
        tstart=period[0],
        # color="hot_r",
        # sort_by_frate=True,
    )
    axraster.plot([1, 2], [50, 50], "k", lw=3)  # lfp scale bar 2.5 mV
    axraster.axis("off")

figure.savefig("raster_example", __file__)
# endregion

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

figure = Fig()
fig, gs = figure.draw(num=1, grid=(2, 2))
fig.suptitle("Change in interspike interval during Sleep Deprivaton")
sessions = subjects.Sd().allsess
for sub, sess in enumerate(sessions):
    sess.trange = np.array([])
    post = sess.epochs.post
    eegSrate = sess.recinfo.lfpSrate
    spikes = sess.spikes.pyr
    sd_period = [post[0], post[0] + 5 * 3600]
    ripples = sess.ripple.events[["start", "end"]].to_numpy()
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

#%% Raster plot before and after onset of recovery sleep
# region
plt.clf()
fig = plt.figure(1, figsize=(10, 15))
gs = gridspec.GridSpec(6, 1, figure=fig)
fig.subplots_adjust(hspace=0.3)
for sub, sess in enumerate(sessions[:6]):
    sess.trange = np.array([])
    post = sess.epochs.post
    period = [post[0] + 4 * 3600, post[0] + 6 * 3600]
    ax = fig.add_subplot(gs[sub])
    sess.spikes.rasterPlot(ax=ax, period=period)
# endregion

#%% firing rate around start of REM (recovery sleep vs control sleep)
# region
plt.clf()
fig = plt.figure(1, figsize=(10, 15))
gs = gridspec.GridSpec(3, 2, figure=fig)
fig.subplots_adjust(hspace=0.3)

for sub, sess in enumerate(sessions):
    sess.trange = np.array([])
    states = sess.brainstates.states
    post = sess.epochs.post
    # instfiring = sess.spikes.instfiring
    spikes = np.concatenate(sess.spikes.pyr)
    if sub < 3:
        rem = states[(states.start > post[0] + 5 * 3600) & (states.name == "rem")]
    else:
        rem = states[(states.start > post[0]) & (states.name == "rem")]

    instf_rem = []
    for epoch in rem.itertuples():
        bins = np.linspace(epoch.start - 5, epoch.start + 5, 11)
        spkcount = np.histogram(spikes, bins=bins)[0] / np.diff(bins)
        instf_rem.append(spkcount)

    mean_frate_rem = np.array(instf_rem).mean(axis=0)
    ax = fig.add_subplot(gs[sub])
    ax.plot(mean_frate_rem)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frate")

# endregion

#%% ratio of firing rate during NREM/REM for recovery sleep vs regular sleep
# region
plt.clf()
fig = plt.figure(1, figsize=(10, 15))
gs = gridspec.GridSpec(3, 2, figure=fig)
fig.subplots_adjust(hspace=0.3)

ax = fig.add_subplot(gs[0])
for sub, sess in enumerate(sessions):
    sess.trange = np.array([])
    states = sess.brainstates.states
    post = sess.epochs.post
    # instfiring = sess.spikes.instfiring
    spikes = np.concatenate(sess.spikes.times)
    if sub < 3:
        rem = states[(states.start > post[0] + 5 * 3600) & (states.name == "rem")]
        nrem = states[(states.start > post[0] + 5 * 3600) & (states.name == "nrem")]
    else:
        rem = states[(states.start > post[0]) & (states.name == "rem")]
        nrem = states[(states.start > post[0]) & (states.name == "nrem")]

    nrem_bins = np.concatenate([[ep.start, ep.end] for ep in nrem.itertuples()])
    nspikes_nrem = np.histogram(spikes, bins=nrem_bins)[0][::2] / nrem.duration

    rem_bins = np.concatenate([[epoch.start, epoch.end] for epoch in rem.itertuples()])
    nspikes_rem = np.histogram(spikes, bins=rem_bins)[0][::2] / rem.duration

    ax.plot(sub, np.mean(nspikes_nrem) / np.mean(nspikes_rem), "*")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frate")

# endregion

#%%* firing rate over SD and recovery sleep of place cells stable from MAZE to POST
# region

spk_all = pd.DataFrame()
sessions = subjects.Sd().allsess
for sub, sess in enumerate(sessions):
    sess.trange = np.array([])
    # maze = sess.epochs.maze
    # pre = sess.epochs.pre
    post = sess.epochs.post
    spks = sess.spikes.times

    # --- stability (firing rate) from maze to end of sleep deprivation ------
    sd_period = [post[0], post[0] + 5 * 3600]
    intervals = sess.utils.getinterval(period=sd_period, nwindows=5)
    sess.spikes.stability.firingRate(periods=intervals)
    stability = sess.spikes.stability.info
    stable_pyr = np.where((stability.q < 4) & (stability.stable == 1))[0]
    pyr = [spks[cell_id] for cell_id in stable_pyr]
    pyr = np.concatenate(pyr)

    # ---- firing rate calculations --------
    # instfiring = sess.spikes.instfiring
    # instfiring_sd = instfiring.loc[
    #     (instfiring.time > post[0]) & (instfiring.time < post[0] + 5 * 3600)
    # ]
    # mean_frate = stats.binned_statistic(
    #     instfiring_sd.time, instfiring_sd.frate, bins=sd_bin
    # )

    binsize = 300  # 5 minutes
    sd_bin = np.arange(post[0], post[0] + 10 * 3600, binsize)
    spkcnt = np.histogram(pyr, bins=sd_bin)[0] / binsize
    norm_spkcnt = spkcnt / np.sum(spkcnt)

    spk_all = spk_all.append(
        pd.DataFrame(
            {
                "time": (sd_bin[:-1] - post[0]) / 3600,
                "spikes": gaussian_filter(norm_spkcnt, sigma=1),
                "subject": sess.recinfo.session.sessionName,
            }
        )
    )

mean_fr = spk_all.groupby("time").mean()

figure = Fig()
fig, gs = figure.draw(num=1, grid=[4, 3])
axfr = plt.subplot(gs[0, 0])
figure.panel_label(axfr, "a")
sns.lineplot(
    x="time",
    y="spikes",
    # hue="subject",
    data=spk_all,
    ax=axfr,
    ci=None,
    style="subject",
    legend=None,
    color="gray",
    ls="--",
    dashes=False,
    alpha=0.5,
)
mean_fr.plot(ax=axfr, color="k", linewidth=2, alpha=0.7, legend=None)
axfr.plot([0, 5], [0.005, 0.005], color="#f14646", linewidth=5)
axfr.text(2.5, 0.0055, "SD", ha="center")
axfr.set_ylabel("Normalized \n spike counts")
axfr.set_xlabel("Time (h)")

# figure.savefig("firing_sd", __file__)
# endregion

#%%* Mean correlaion across POST
# region

mean_corr_all = pd.DataFrame()
sessions = subjects.Sd().allsess + subjects.Nsd().allsess
for sub, sess in enumerate(sessions):
    sess.trange = np.array([])
    post = sess.epochs.post
    spks = sess.spikes.times

    # --- stability (firing rate) from maze to end of sleep deprivation ------
    sd_period = [post[0], post[0] + 5 * 3600]
    intervals = sess.utils.getinterval(period=sd_period, nwindows=5)
    sess.spikes.stability.firingRate(periods=intervals)
    stability = sess.spikes.stability.info
    stable_pyr = np.where((stability.q < 4) & (stability.stable == 1))[0]
    pyr = [spks[cell_id] for cell_id in stable_pyr]

    pyr = sess.spikes.pyr

    window_sz = 600
    slideby = 300
    windows = np.arange(post[0], post[0] + 8 * 3600 - window_sz, slideby)
    mean_corr = []
    mean_frate = []
    for start in windows:
        corr = sess.spikes.corr.pairwise(pyr, period=[start, start + window_sz])
        frate = sess.spikes.firing_rate(pyr, period=[start, start + window_sz])
        mean_corr.append(np.nanmean(corr))
        mean_frate.append(np.mean(frate))

    mean_corr_all = mean_corr_all.append(
        pd.DataFrame(
            {
                "time": (windows - post[0] + window_sz / 2) / 3600,
                "corr": gaussian_filter1d(mean_corr / np.sum(mean_corr), sigma=1),
                "frate": gaussian_filter1d(mean_frate / np.sum(mean_frate), sigma=1),
                "subject": sess.recinfo.session.sessionName,
                "grp": sess.recinfo.animal.tag,
            }
        )
    )


figure = Fig()
fig, gs = figure.draw(num=1, grid=[4, 3], hspace=0.3)
group = ["sd", "nsd"]
for i, grp in enumerate(group):
    grp_data = mean_corr_all[mean_corr_all.grp == grp]
    mean_corr = grp_data.groupby("time").mean()

    axfr = plt.subplot(gs[i, :-1])
    figure.panel_label(axfr, "a")
    sns.lineplot(
        x="time",
        y="corr",
        # hue="subject",
        data=grp_data,
        ax=axfr,
        ci=None,
        style="subject",
        legend=None,
        color="gray",
        ls="--",
        dashes=False,
        alpha=0.5,
    )
    # mean_corr.plot(ax=axfr, y="frate", color="k", linewidth=2, alpha=0.7, legend=None)
    mean_corr.plot(ax=axfr, y="corr", color="k", linewidth=2, alpha=0.7, legend=None)
    axfr.plot([0, 5], [0.005, 0.005], color="#f14646", linewidth=5)
    axfr.text(2.5, 0.0055, "SD", ha="center")
    axfr.set_ylabel("Mean correlation")
    axfr.set_xlabel("Time since ZT0 (h)")
    axfr.set_ylim([0, 0.02])
    axfr.ticklabel_format(axis="y", scilimits=(0, 0))

figure.savefig("meancorr_sd_nsd", __file__)
# endregion


#%% MUA activity during high gamma (slow/fast) periods
# region
"""Did not fing any interesting difference 
"""

for sub, sess in enumerate(sessions[5:6]):

    eegSrate = sess.recinfo.lfpSrate
    maze = sess.epochs.maze
    instfiring = sess.spikes.instfiring
    instfiring = instfiring[(instfiring.time > maze[0]) & (instfiring.time < maze[1])]

    lfpmaze = sess.recinfo.geteeg(chans=sess.theta.bestchan, timeRange=maze)
    peakgamma_periods = sess.gamma.get_peak_intervals(lfpmaze, band=(25, 50))
    peakgamma_periods = (peakgamma_periods / eegSrate) + maze[0]

    gamma_frate = []
    for epoch in peakgamma_periods:
        gamma_frate.extend(
            instfiring[
                (instfiring.time > epoch[0]) & (instfiring.time < epoch[1])
            ].frate
        )

# endregion

#%% Theta phase preference during REM sleep
# region
figure = Fig()
fig, gs = figure.draw(grid=(7, 10))
sessions = subjects.sd([3])
for sub, sess in enumerate(sessions):
    maze = sess.epochs.maze1
    states = sess.brainstates.states
    rems = states[(states.name == "rem") & (states.duration > 5)]
    pyr = sess.spikes.pyr

    phase_all = []
    for cell in pyr:
        phase_cell = []
        for rem in rems.itertuples():
            spks_rem = cell[(cell > rem.start) & (cell < rem.end)]
            lfp = sess.recinfo.geteeg(chans=135, timeRange=[rem.start, rem.end])
            t = np.linspace(rem.start, rem.end, len(lfp))
            thetaparam = sess.theta.getParams(lfp)

            phase_cell.extend(np.interp(spks_rem, t, thetaparam.angle))
        phase_all.append(phase_cell)

    angle_bin = np.linspace(0, 360, 37)
    angle_center = angle_bin[:-1] + np.mean(np.diff(angle_bin)) / 2
    phase_hist_df = pd.DataFrame()
    phase_hist_df["phase"] = angle_center
    for cell_id, cell in enumerate(phase_all):
        histcell = np.histogram(cell, bins=angle_bin)[0]
        phase_hist_df[cell_id] = histcell  # / np.sum(histcell)
        phase_prefer = angle_center[np.argmax(histcell)]

        ax = plt.subplot(gs[cell_id], projection="polar")
        ax.bar(phase_hist_df["phase"], histcell)
        # ax.plot(cell, ".")


# endregion
