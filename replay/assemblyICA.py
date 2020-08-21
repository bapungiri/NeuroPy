#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter

from mathutil import getICA_Assembly
from callfunc import processData

basePath = [
    # "/data/Clustering/SleepDeprivation/RatJ/Day1/",
    # "/data/Clustering/SleepDeprivation/RatK/Day1/",
    # "/data/Clustering/SleepDeprivation/RatN/Day1/",
    # "/data/Clustering/SleepDeprivation/RatN/Day1/",
    # "/data/Clustering/SleepDeprivation/RatJ/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day2/",
    # "/data/Clustering/SleepDeprivation/RatN/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day4/"
    "/data/Clustering/SleepDeprivation/RatN/Day4/"
]


sessions = [processData(_) for _ in basePath]

#%% Reactivation strength
plt.clf()
fig = plt.figure(1, figsize=(10, 15))
gs = gridspec.GridSpec(16, 1, figure=fig)
fig.subplots_adjust(hspace=0.3)

for sub, sess in enumerate(sessions):
    sess.trange = np.array([])
    maze = sess.epochs.maze
    post = sess.epochs.post
    activation = sess.replay.assemblyICA.detect(template=maze, match=post)

    for i in range(16):
        ax = fig.add_subplot(gs[i])
        ax.plot(activation[i])

# sess.brainstates.detect()

# violations = sess.spikes.stability.violations
#%% Reactivation strength vs delta amplitude
# region
plt.clf()
fig = plt.figure(1, figsize=(10, 15))
gs = gridspec.GridSpec(1, 1, figure=fig)
fig.subplots_adjust(hspace=0.3)

for sub, sess in enumerate(sessions):
    sess.trange = np.array([])

    maze = sess.epochs.maze
    post = sess.epochs.post
    activation, t = sess.replay.assemblyICA.detect(template=maze, match=post)

    swa = sess.swa.time


# endregion
