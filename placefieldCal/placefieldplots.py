#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter
import subjects
from callfunc import processData


#%% 1D place field in openfield arena
# region
plt.close("all")
for sub, sess in enumerate(sessions):

    sess.trange = np.array([])

    sess.placefield.pf1d.compute()
    sess.placefield.pf1d.plot(pad=0.5, normalize=True)
    # sess.placefield.pf1d.plotRaw()


#%% 2D place field in openfield arena
# region
plt.close("all")
for sub, sess in enumerate(sessions):

    sess.trange = np.array([])

    # sess.spikes.fromCircus(fileformat="same_folder")
    sess.placefield.pf2d.compute(gridbin=10)
    sess.placefield.pf2d.plotMap()
    # sess.placefield.pf2d.plotRaw()
    # sess.position.export2Neuroscope()
#     sess.spikes.stability.firingRate()


# stability = sess.spikes.stability.isStable
# unstable = sess.spikes.stability.unstable
# stable = sess.spikes.stability.stable

# endregion

#%% Theta phase precession Open field
# region
"""Calculating theta precession in open field experiments, but abondoned for now
"""
plt.clf()
fig = plt.figure(1, figsize=(10, 15))
gs = gridspec.GridSpec(8, 5, figure=fig)
fig.subplots_adjust(hspace=0.3)
for sub, sess in enumerate(sessions):

    sess.trange = np.array([])

    # sess.spikes.fromCircus(fileformat="same_folder")
    sess.placefield.pf2d.compute()
    sess.placefield.pf2d.plotMap()
    sess.placefield.pf2d.plotRaw()
    maps = sess.placefield.pf2d.maps
    for cell, pfmap in enumerate(maps):
        pfmap_linear = np.ravel(pfmap)
        ax = fig.add_subplot(gs[cell])
        ax.plot(pfmap_linear)


# endregion

#%%  Plot place fields
# region
sessions = subjects.sd([3])
for sub, sess in enumerate(sessions):
    period = sess.epochs.maze1
    sess.placefield.pf2d.compute(period=period, speed_thresh=10)
    sess.placefield.pf2d.plotRaw(subplots=(10, 8), speed_thresh=False)

    period = sess.epochs.maze2
    sess.placefield.pf2d.compute(period=period, speed_thresh=5)
    sess.placefield.pf2d.plotRaw(subplots=(10, 8), speed_thresh=False)


# endregion