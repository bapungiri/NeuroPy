#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter

from callfunc import processData

basePath = [
    "/data/Clustering/SleepDeprivation/RatJ/Day1/",
    "/data/Clustering/SleepDeprivation/RatK/Day1/",
    "/data/Clustering/SleepDeprivation/RatN/Day1/",
    # "/data/Clustering/SleepDeprivation/RatJ/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day2/",
    # "/data/Clustering/SleepDeprivation/RatN/Day2/",
]


sessions = [processData(_) for _ in basePath]


#%% Stabiliy of cells using firing rate
plt.clf()
fig = plt.figure(1, figsize=(10, 15))
gs = gridspec.GridSpec(3, 1, figure=fig)
fig.subplots_adjust(hspace=0.3)
fig.suptitle("Change in interspike interval during Sleep Deprivaton")

for sub, sess in enumerate(sessions):

    sess.trange = np.array([])
    spikes = sess.spikes.times
    pre = sess.epochs.pre
    post = sess.epochs.post
    spkinfo = sess.spikes.info
    reqcells_id = np.where(spkinfo["q"] < 4)[0]
    # spikes = [spikes[cell] for cell in reqcells_id]

    first_hour = pre[]

    frate_pre = [np.histogram(cell, bins=pre)[0] / np.diff(pre) for cell in spikes]
    frate_post = [np.histogram(cell, bins=post)[0] / np.diff(post) for cell in spikes]

    ax = fig.add_subplot(gs[sub])
    ax.plot(frate_pre, frate_post, ".")
    ax.plot([0, max(frate_pre)], [0, max(frate_pre)])
    # ax.axis("equal")

