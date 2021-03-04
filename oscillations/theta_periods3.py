# %%
from typing import Dict

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy
import pingouin as pg
import scipy.signal as sg
import scipy.stats as stats
import seaborn as sns
import signal_process
import subjects
from mathutil import threshPeriods
from plotUtil import Colormap, Fig
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from joblib import Parallel, delayed
import networkx as nx
from sklearn.cluster import spectral_clustering
import scipy.cluster.hierarchy as sch
import subjects
import scipy.fft as fft

#%% Scratchpad for selecting correlated channels
# region
# figure = Fig()
# fig, gs = figure.draw(num=1, grid=(4, 4))

# sessions = subjects.Sd().allsess[:3] + subjects.Nsd().allsess[:3]
# sessions = subjects.Of().ratNday4
sessions = subjects.Nsd().ratNday2
for sub, sess in enumerate(sessions):
    maze = sess.epochs.maze
    eegSrate = sess.recinfo.lfpSrate
    ca1_chans = sess.ripple.bestchans
    nShanks = sess.recinfo.nShanks
    chan_grp = sess.recinfo.goodchangrp
    chan_grp = [_ for _ in chan_grp if _]
    chans = np.array(chan_grp[-1])
    chans = sess.recinfo.goodchans

    lfp = np.asarray(sess.recinfo.geteeg(chans=chans, timeRange=maze))
    corr_chans = np.corrcoef(lfp)
    corr_chans = np.where(corr_chans > 0.7, 1, 0)
    np.fill_diagonal(corr_chans, 0)

    # pair_indices = np.tril_indices_from(corr_chans, k=-1)
    # vals = corr_chans[pair_indices]
    # uncorr_pairs = np.where(vals == 0)[0]
    # uncorr_chans = (
    #     chans[pair_indices[0][uncorr_pairs]],
    #     chans[pair_indices[1][uncorr_pairs]],
    # )

    # grp1 = np.unique(uncorr_chans[0])
    # grp2 = np.unique(uncorr_chans[1])

    # G = nx.from_numpy_array(corr_chans)
    # nx.draw(G)
    # nx.clustering(G)
    # nx.closeness_centrality(G)
    # a = nx.subgraph(G, nbunch=2)
    # nx.node_degree_xy(G)
    # # plt.savefig("path_graph1.png")
    # plt.show()
    # sc = SpectralClustering(2, affinity="precomputed", n_init=100)
    # sc.fit(corr_chans)

    # a = np.where(sc.labels_ == 1)[0]

    # grid = np.ix_(a, a)
    # corr_new = corr_chans[grid]
    labels = spectral_clustering(affinity=corr_chans, n_clusters=2)

    # vector of ('55' choose 2) pairwise distances
    d = sch.distance.pdist(corr_chans)
    indices = np.triu_indices_from(corr_chans, k=1)
    L = sch.linkage(corr_chans[indices], method="complete")
    ind = sch.fcluster(L, 0.7, "distance")
    sort_ind = np.argsort(labels)
    corr_new = corr_chans[np.ix_(sort_ind, sort_ind)]
    # sns.heatmap(corr_new)
    plt.imshow(corr_new)

# endregion


#%% Gamma around theta for correlated channels using wavelet
# region
figure = Fig()
fig, gs = figure.draw(num=1, grid=(4, 4))

# sessions = subjects.Sd().allsess[:3] + subjects.Nsd().allsess[:3]
# sessions = subjects.Sd().ratJday1
# sessions = subjects.Nsd().ratNday2
sessions = subjects.Of().ratNday4
for sub, sess in enumerate(sessions):
    maze = sess.epochs.maze
    eegSrate = sess.recinfo.lfpSrate
    ca1_chans = sess.ripple.bestchans
    nShanks = sess.recinfo.nShanks
    chan_grp = sess.recinfo.goodchangrp
    chan_grp = [_ for _ in chan_grp if _]
    chans = np.array(chan_grp[-1])
    chans = sess.recinfo.goodchans

    lfp = np.asarray(sess.recinfo.geteeg(chans=chans, timeRange=maze))
    corr_chans = np.corrcoef(lfp)
    corr_chans = np.where(corr_chans > 0.7, 1, 0)
    np.fill_diagonal(corr_chans, 0)

    labels = spectral_clustering(affinity=corr_chans, n_clusters=2)
    sort_ind = np.argsort(labels)
    corr_new = corr_chans[np.ix_(sort_ind, sort_ind)]

    grp1 = chans[np.where(labels == 0)[0]]
    grp2 = chans[labels == 1]

    gamma_all = []
    for grp_chans in [grp1, grp2]:
        grp_ind = np.intersect1d(chans, grp_chans, return_indices=True)[1]
        lfp_ca1 = np.median(lfp[grp_ind, :], axis=0)
        # plt.plot(lfp_ca1)
        strong_theta = sess.theta.getstrongTheta(
            lfp_ca1, lowthresh=0.1, highthresh=0.5
        )[0]

        # --- phase estimation by waveshape --------

        theta_param = signal_process.ThetaParams(
            strong_theta, fs=1250, method="waveshape"
        )
        lfp_angle = theta_param.angle

        frgamma = np.arange(25, 150)
        # ----- wavelet power for gamma oscillations----------
        wavdec = signal_process.wavelet_decomp(
            strong_theta, freqs=frgamma, sampfreq=eegSrate
        )
        wav = wavdec.colgin2009()
        wav = stats.zscore(wav, axis=1)

        # ----segmenting gamma wavelet at theta phases ----------
        bin_angle = np.linspace(0, 360, int(360 / 9) + 1)
        phase_centers = bin_angle[:-1] + np.diff(bin_angle).mean() / 2

        bin_ind = np.digitize(lfp_angle, bin_angle)

        gamma_at_theta = pd.DataFrame()
        for i in np.unique(bin_ind):
            find_where = np.where(bin_ind == i)[0]
            gamma_at_theta[bin_angle[i - 1]] = np.mean(wav[:, find_where], axis=1)
        gamma_at_theta.insert(0, column="freq", value=frgamma)
        gamma_at_theta.set_index("freq", inplace=True)

        gamma_all.append(gamma_at_theta)

    for i, data in enumerate(gamma_all):
        ax = plt.subplot(gs[i])
        ax.contourf(
            phase_centers,
            frgamma,
            np.asarray(data),
            cmap="jet",
            levels=50,
            # origin="lower",
        )
        ax.set_xticks([0, 90, 180, 270, 360])
        ax.set_yticks(np.arange(25, 150, 30))
        ax.set_title(f"Shank {i+1}")


# endregion

#%% Gamma around theta for correlated channels using phase extraction
# region
figure = Fig()
fig, gs = figure.draw(num=1, grid=(4, 4))

# sessions = subjects.Sd().allsess[:3] + subjects.Nsd().allsess[:3]
# sessions = subjects.Sd().ratJday1
# sessions = subjects.Nsd().ratNday2
sessions = subjects.Of().ratNday4
for sub, sess in enumerate(sessions):
    maze = sess.epochs.maze
    eegSrate = sess.recinfo.lfpSrate
    ca1_chans = sess.ripple.bestchans
    nShanks = sess.recinfo.nShanks
    chan_grp = sess.recinfo.goodchangrp
    chan_grp = [_ for _ in chan_grp if _]
    chans = sess.recinfo.goodchans

    lfp = np.asarray(sess.recinfo.geteeg(chans=chans, timeRange=maze))
    corr_chans = np.corrcoef(lfp)
    corr_chans = np.where(corr_chans > 0.7, 1, 0)
    np.fill_diagonal(corr_chans, 0)

    labels = spectral_clustering(affinity=corr_chans, n_clusters=2)
    sort_ind = np.argsort(labels)
    corr_new = corr_chans[np.ix_(sort_ind, sort_ind)]

    grp1 = chans[np.where(labels == 0)[0]]
    grp2 = chans[labels == 1]

    gamma_all = []
    for grp_chans in [grp1, grp2]:
        grp_ind = np.intersect1d(chans, grp_chans, return_indices=True)[1]
        lfp_ca1 = np.median(lfp[grp_ind, :], axis=0)
        # plt.plot(lfp_ca1)
        strong_theta = sess.theta.getstrongTheta(lfp_ca1, lowthresh=0, highthresh=1)[0]

        # --- phase estimation by waveshape --------

        theta_param = signal_process.ThetaParams(
            strong_theta, fs=1250, method="waveshape"
        )
        lfp_angle = theta_param.angle
        gamma = signal_process.filter_sig.highpass(strong_theta, cutoff=60)
        gamma_at_theta, _, angle_centers = theta_param.break_by_phase(
            gamma, binsize=30, slideby=9
        )

        df = pd.DataFrame()
        f_ = None
        for lfp_, center in zip(gamma_at_theta, angle_centers):
            f_, pxx = sg.welch(lfp_, nperseg=2 * 1250, noverlap=625, fs=1250)
            df[center] = pxx

        df.insert(0, "freq", f_)

        gamma_all.append(df)

    for i, df in enumerate(gamma_all):
        ax = plt.subplot(gs[i])
        data = df[df.freq < 200].set_index("freq").transform(stats.zscore, axis=1)
        ax.pcolormesh(
            data.columns,
            data.index,
            gaussian_filter(data, sigma=1),
            cmap="jet",
            shading="auto",
        )
        ax.set_xlabel(r"$\theta$ phase")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_xticks(np.linspace(0, 360, 5))
        # ax.set_title(bin_names[i])


# endregion


#%% Gamma bouts and cross-correlogram of pyramidal/interneurons
# region
figure = Fig()
fig, gs = figure.draw(num=1, grid=(12, 5))
sessions = subjects.Of().ratNday4
for sub, sess in enumerate(sessions):

    eegSrate = sess.recinfo.lfpSrate
    maze = sess.epochs.maze
    spikes = sess.spikes.pyr  # sess.spikes.intneur
    chan = 12

    lfp = sess.recinfo.geteeg(chans=chan, timeRange=maze)

    def get_ccg_gamma(low, high):
        gamma_filt = signal_process.filter_sig.bandpass(
            lfp, lf=low, hf=high, fs=eegSrate
        )
        hilbert_sig = signal_process.hilbertfast(gamma_filt)
        hilbert_amp = np.abs(hilbert_sig)
        bouts = threshPeriods(
            stats.zscore(hilbert_amp),
            lowthresh=1,
            highthresh=1.1,
            minDuration=10,
            minDistance=10,
        )
        bouts = bouts / eegSrate + maze[0]

        gamma_spikes = []
        for cell in spikes:
            spike_ind = np.digitize(cell, bins=np.ravel(bouts))
            gamma_spikes.append(cell[spike_ind % 2 == 1])

        acg = np.asarray(sess.spikes.get_acg(spikes=gamma_spikes, window_size=0.1))
        acg_norm = stats.zscore(acg, axis=1)

        return acg

    acg_slgamma = get_ccg_gamma(25, 55)
    acg_medgamma = get_ccg_gamma(60, 100)

    t = np.linspace(-1, 1, 101) * 50
    i = 0
    for sl, med in zip(acg_slgamma, acg_medgamma):

        if np.max(sl) > 5 and np.max(med) > 5:
            gs_ = figure.subplot2grid(gs[i], grid=(1, 2))
            ax1 = plt.subplot(gs_[0])
            ax1.fill_between(t, sl)

            ax2 = plt.subplot(gs_[1], sharex=ax1, sharey=ax1)
            ax2.fill_between(t, med)
            i += 1

    # plt.imshow(acg_norm, cmap="tab20c")
    # acg_half = acg_norm[:, 28:]
    # sum_acg = np.sum(acg_half, axis=1)
    # zero_ccg_ind = np.where(sum_acg == 0)[0]
    # acg_half = np.delete(acg_half, zero_ccg_ind, axis=0)

    # # sort_ind = np.argsort(acg_norm, axis=1)
    # # acg_sorted = acg_half[sort_ind, :]
    # # plt.imshow(acg_half, cmap="tab20c")

    # acg_new = np.zeros_like(acg_half)
    # for i, vals in enumerate(acg_half):
    #     if np.count_nonzero(vals) > 0:
    #         peak = sg.find_peaks(vals)[0]
    #         vals[peak] = 1
    #         vals[np.setdiff1d(np.arange(len(vals)), peak)] = 0
    #         acg_new[i, :] = vals

    # sort_ind = np.argsort(np.argmax(acg_new, axis=1))
    # acg_new = acg_new[sort_ind, :]

    # labels = spectral_clustering(affinity=a, n_clusters=2)
    # sort_ind = np.argsort(labels)
    # corr_new = corr_chans[np.ix_(sort_ind, sort_ind)]

    # pdist = sch.distance.pdist(a)
    # linkage = sch.linkage(pdist, method="complete")
    # idx = sch.fcluster(linkage, 0.5 * pdist.max(), "distance")


# endregion

#%% Summation of shifted sinousoids
# region
figure = Fig()
fig, gs = figure.draw(num=1, grid=(3, 1), size=(14, 6))
t = np.linspace(0, 2, 1000)
y1 = np.sin(2 * np.pi * 8 * t) + np.sin(2 * np.pi * 16 * t)
y2 = np.sin(2 * np.pi * 7 * t + 0.5)
y3 = 0.2 * np.sin(2 * np.pi * 10 * t + 0.8)
y4 = sg.sawtooth(2 * np.pi * 8 * t)[::-1]
y_sum = y1 + y2 + y3 + y4

ax = plt.subplot(gs[0])
ax.plot(y1)
ax.plot(y2)
ax.plot(y3)
ax.plot(y4)

ax = plt.subplot(gs[1], sharex=ax)
ax.plot(y_sum)


# endregion

#%% check whitening
# region
figure = Fig()
fig, gs = figure.draw(num=1, grid=(2, 2))
sessions = subjects.Of().ratNday4

for sub, sess in enumerate(sessions):
    maze = sess.epochs.maze
    eegSrate = sess.recinfo.lfpSrate
    lfp = sess.recinfo.geteeg(chans=12, timeRange=maze)

    f, pxx = sg.welch(lfp, fs=eegSrate, nperseg=5 * 1250, noverlap=2 * 1250)

    ax = plt.subplot(gs[0])
    ax.plot(f, pxx)
    ax.set_yscale("log")
    ax.set_xscale("log")

    pxx_smooth = gaussian_filter1d(pxx, sigma=100)

    ax.plot(f, pxx_smooth)

# endregion

#%% Check theta across layers during gamma events
# region
figure = Fig()
fig, gs = figure.draw(num=1, grid=(2, 2))
sessions = subjects.Of().ratNday4
for sub, sess in enumerate(sessions):
    eegSrate = sess.recinfo.lfpSrate
    maze = sess.epochs.maze

    lfp = sess.recinfo.geteeg(chans=47, timeRange=maze)
    gamma_filt = signal_process.filter_sig.bandpass(lfp, lf=60, hf=100, fs=eegSrate)
    hilbert_sig = signal_process.hilbertfast(gamma_filt)
    hilbert_amp = np.abs(hilbert_sig)
    bouts = threshPeriods(
        stats.zscore(hilbert_amp),
        lowthresh=1,
        highthresh=1.1,
        minDuration=10,
        minDistance=10,
    )
    bouts = bouts / eegSrate + maze[0]

    shank_chans = sess.recinfo.goodchangrp[2]
    bout_ind = 0
    period = [bouts[bout_ind][0] - 1, bouts[bout_ind][1] + 1]
    lfp_gamma = np.asarray(sess.recinfo.geteeg(chans=shank_chans, timeRange=period))

    concat_gamma_lfp = []
    for bout in bouts:
        concat_gamma_lfp.append(
            np.asarray(sess.recinfo.geteeg(chans=shank_chans, timeRange=bout))
        )

    concat_gamma_lfp = np.hstack(concat_gamma_lfp)

    corr_chans = np.corrcoef(concat_gamma_lfp)

    # lfp_gamma = lfp_gamma.T + np.linspace(1200, 0, len(shank_chans))

# endregion

#%% Different gamma band amplitude around reactivation
# region
figure = Fig()
fig, gs = figure.draw(num=1, grid=(4, 5), size=(12, 6))
sessions = subjects.Of().ratNday4
for sub, sess in enumerate(sessions):
    maze = sess.epochs.maze
    pre = sess.epochs.pre
    post = sess.epochs.post
    sprinkle = sess.epochs.sprinkle
    pre_sprinkle = [maze[0], sprinkle[0]]

    x = sess.position.x
    y = sess.position.y
    t = sess.position.t

    maze_indx = np.where((t > maze[0]) & (t < maze[1]))[0]
    maze_x = x[maze_indx]
    maze_y = y[maze_indx]
    xbins = np.linspace(np.min(maze_x), np.max(maze_x), 40)
    ybins = np.linspace(np.min(maze_y), np.max(maze_y), 40)
    xgrid, ygrid = np.meshgrid(xbins, ybins)

    sess.replay.assemblyICA.getAssemblies(period=pre_sprinkle)
    activation_maze, t_act = sess.replay.assemblyICA.getActivation(period=maze)
    t_act = t_act[:-1]

    act_zscore = stats.zscore(activation_maze, axis=1)
    act_zscore = np.where(act_zscore > 2, 1, 0)


# endregion
