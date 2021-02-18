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


#%% Gamma around theta for correlated channels
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

    labels = spectral_clustering(affinity=corr_chans, n_clusters=3)
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
