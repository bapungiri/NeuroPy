#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from pathlib import Path
import matplotlib as mpl
from plotUtil import Colormap
import scipy.signal as sg
from ccg import correlograms
from plotUtil import Fig
from sklearn.cluster import KMeans
import subjects

#%% Plotting cell statistics
# region
plt.clf()
fig = plt.figure(1, figsize=(10, 15))
gs = gridspec.GridSpec(3, 3, figure=fig)
fig.subplots_adjust(hspace=0.3)
for sub, sess in enumerate(sessions):
    sess.trange = np.array([])
    # sess.spikes.stability.firingRate()
    stability = sess.spikes.stability.info
    stability.loc[stability.q < 4, "cellType"] = "pyr"
    stability.loc[stability.q == 6, "cellType"] = "mua"
    stability.loc[stability.q == 8, "cellType"] = "intneur"

    ax = fig.add_subplot(gs[sub])
    # stability.plot.bar(x="cellType", y="stable", stacked=True, ax=ax, rot=0)
    sns.countplot(
        x="cellType",
        hue="stable",
        data=stability,
        order=["pyr", "intneur", "mua"],
        ax=ax,
        palette="Pastel1",
    )

# endregion

#%% Auto labelling of cells scratchpad
# region
figure = Fig()
fig, gs = figure.draw(num=1, grid=(1, 1))
sessions = subjects.Sd().ratNday1
for sub, sess in enumerate(sessions):
    spikes = sess.spikes.times
    sess.spikes.label_celltype()
    ccgs = sess.spikes.get_acg(spikes=spikes)
    ccg_width = ccgs[0].shape[-1]
    ccg_right = [_[26:] for _ in ccgs]
    burstiness = np.asarray([len(ccg) / np.sum(ccg) for ccg in ccg_right])

    period_dur = sess.recinfo.getNframesEEG / 1250
    templates = sess.spikes.templates

    frate = np.asarray([len(cell) / period_dur for cell in spikes])
    waveform = np.asarray(
        [cell[np.argmax(np.ptp(cell, axis=1)), :] for cell in templates]
    )

    n_t = waveform.shape[1]
    center = np.int(n_t / 2)
    left_peak = np.max(waveform[:, :center], axis=1)
    right_peak = np.max(waveform[:, center + 1 :], axis=1)
    peak_ratio = left_peak / right_peak

    isi = [np.diff(_) for _ in spikes]
    isi_bin = np.arange(0, 0.1, 0.001)
    isi_hist = np.asarray([np.histogram(_, bins=isi_bin)[0] for _ in isi])
    n_spikes_ref = np.sum(isi_hist[:, :2], axis=1) + 1e-16
    ref_period_ratio = np.max(isi_hist, axis=1) / n_spikes_ref

    sum_peak = np.asarray([np.max(ccg[20:24]) for ccg in ccgs])
    sum_refractory = np.asarray([np.sum(ccg[24:26]) for ccg in ccgs]) + 1e-16
    ref_ratio = sum_peak / sum_refractory

    param1 = peak_ratio
    param2 = burstiness  # np.log10(sum_peak / sum_refractory)
    param3 = frate

    features = np.vstack((param1, param2, param3)).T
    kmeans = KMeans(n_clusters=2).fit(features)
    y_means = kmeans.predict(features)
    ax = plt.subplot(gs[0], projection="3d")
    # ax.scatter(param3, param2, param1)
    ax.scatter(param1, param2, param3, c=y_means, s=50, cmap="viridis")

    # plt.plot(np.log10(frate), np.log(sum_peak / sum_refractory), ".")

    # for cell_id, ccg in enumerate(ccgs):
    #     ax = plt.subplot(gs[cell_id])
    #     ax.bar(np.arange(len(ccg)), ccg)


# endregion

#%% Plot autocorrelograms
# region
figure = Fig()
fig, gs = figure.draw(num=1, grid=(10, 11))
sessions = subjects.Sd().ratNday1
for sub, sess in enumerate(sessions):
    # sess.recinfo.sampfreq = 30000
    sess.spikes.label_celltype()
    spikes = sess.spikes.intneur
    duration = sess.recinfo.getNframesEEG / 1250
    frate = [len(_) / np.ptp(_) for _ in spikes]
    ccgs = sess.spikes.get_acg(spikes=spikes)
    for cell_id, ccg in enumerate(ccgs):
        ax = plt.subplot(gs[cell_id])
        ax.bar(np.arange(len(ccg)), ccg)
        ax.set_title(f"{np.round(frate[cell_id],2)}")
        ax.axis("off")


# endregion
#%% Plot spike amplitude over time and CCG
# region
for sub, sess in enumerate(sessions):
    sess.trange = np.array([])

# endregion
