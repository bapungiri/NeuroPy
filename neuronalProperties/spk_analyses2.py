#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter
from ccg import correlograms
from mathutil import getICA_Assembly
from callfunc import processData
import subjects
from plotUtil import Fig
import networkx as nx
import seaborn as sns

#%% Network analysis of sleep deprivation
# region

figure = Fig()
fig, gs = figure.draw(num=1, grid=(2, 2))
sessions = subjects.Sd().ratSday3
for sub, sess in enumerate(sessions):
    period = sess.epochs.sd
    sd_windows = np.arange(period[0], period[1] + 3600, 3600)
    pyr = sess.spikes.pyr

    for window in sd_windows:
        bins = np.arange(window, window + 3600, 0.25)
        spk_counts = np.asarray([np.histogram(cell, bins=bins)[0] for cell in pyr])
        corr = np.corrcoef(spk_counts)


# endregion

#%% Early to late novel exploration change in pairwise correlation
# region

figure = Fig()
fig, gs = figure.draw(num=1, grid=(2, 2))
sessions = (
    subjects.Sd().allsess
    + subjects.Nsd().allsess
    + subjects.Of().ratNday4
    + subjects.Tn().ratSday5
)
df = pd.DataFrame()
for sub, sess in enumerate(sessions):
    spks = sess.spikes.pyr
    try:
        maze = sess.epochs.maze1
    except:
        maze = sess.epochs.maze

    # maze_early = [maze[0], maze[0] + 15 * 60]
    # maze_late = [maze[-1] - 15 * 60, maze[-1]]
    # pcorr_early = sess.spikes.corr.pairwise(spikes=spks, period=maze_early)
    # pcorr_late = sess.spikes.corr.pairwise(spikes=spks, period=maze_late)
    # ax = plt.subplot(gs[0])
    # ax.scatter(pcorr_early, pcorr_late, s=1)
    # ax.axline((0, 0), (1, 1))
    # corr_across_time, time = sess.spikes.corr.across_time_window(
    #     spikes=spks, period=maze
    # )

    maze_parts = np.arange(maze[0], maze[1] + 1)
    slide_views = np.lib.stride_tricks.sliding_window_view(maze_parts, 600)[::100, :]

    pcorr = []
    for window in slide_views:
        bins = np.arange(window[0], window[-1], 0.25)
        spk_cnts = np.asarray([np.histogram(cell, bins=bins)[0] for cell in spks])
        corr = np.corrcoef(spk_cnts)
        pcorr.append(corr[np.tril_indices_from(corr, k=-1)])
    pcorr = np.asarray(pcorr).T
    pcorr = pcorr[~np.isnan(pcorr).any(axis=1)]

    nQuantiles = 5
    quantiles = pd.qcut(pcorr[:, 0], q=nQuantiles, labels=False)
    for quant_id in range(nQuantiles):
        indx = np.where(quantiles == quant_id)[0]
        mean_corr = np.mean(pcorr[indx, :], axis=0)
        # ax.plot(mean_corr, color="k", alpha=0.3 + quant_id / 7)
        df = df.append(
            pd.DataFrame(
                {
                    "quantiles": [quant_id + 1] * 2,
                    "mean_corr": mean_corr[[0, -1]],
                    "period": ["first 10 minutes", "last 10 minutes"],
                }
            )
        )

ax = plt.subplot(gs[0])
sns.barplot(data=df, x="quantiles", y="mean_corr", hue="period", ax=ax, ci="sd")
ax.set_xlabel("quantiles")
ax.set_ylabel("mean correlation")
ax.set_title("Comapring change in mean correlation for low to highly correlated pairs")
figure.savefig("mean_corr_change_quantiles")
# endregion
