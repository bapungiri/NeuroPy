#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter1d, gaussian_filter
import subjects
from plotUtil import Fig
from mathutil import threshPeriods, hmmfit1d
from typing import Annotated

#%% 1D place field in openfield arena
# region
sessions = subjects.openfield()
for sub, sess in enumerate(sessions):

    sess.trange = np.array([])

    sess.placefield.pf1d.compute()
    sess.placefield.pf1d.plot(pad=0.5, normalize=True)
    # sess.placefield.pf1d.plotRaw()
# endregion

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

#%%  Plot 2D place fields in linear type tracks
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

#%%  Plot 1D place fields in linear type tracks
# region
figure = Fig()
fig, gs = figure.draw(num=1, grid=(1, 2))
sessions = subjects.sd([3])
for sub, sess in enumerate(sessions):
    # period = sess.epochs.maze1
    ax = plt.subplot(gs[0])
    sess.placefield.pf1d.compute(track_name="maze1", run_dir="foward")
    sess.placefield.pf1d.plot(ax=ax, speed_thresh=True, normalize=True)
    # sess.placefield.pf1d.plot_raw(speed_thresh=True, subplots=None)
    ax.set_title("Maze1")

    ratemaps = sess.placefield.pf1d.no_thresh["ratemaps"]
    cell_order = np.argsort(np.argmax(np.asarray(ratemaps), axis=1))

    ax = plt.subplot(gs[1])
    sess.placefield.pf1d.compute(track_name="maze2", run_dir="forward")
    sess.placefield.pf1d.plot(
        ax=ax, speed_thresh=True, normalize=True, sortby=cell_order
    )
    ax.set_title("Maze2")
    # sess.placefield.pf1d.plot_raw(speed_thresh=True, subplots=None)

# endregion

#%% Theta-precession in linear type tracks
# region
figure = Fig()
fig, gs = figure.draw(num=1, grid=(4, 2), size=(5, 12))
sessions = subjects.sd([3])
for sub, sess in enumerate(sessions):
    track_name = "maze1"
    maze = sess.epochs[track_name]
    sess.placefield.pf1d.compute(track_name, run_dir="forward", grid_bin=5, smooth=1)
    # sess.placefield.pf1d.plot(normalize=True)
    sess.placefield.pf1d.phase_precession(theta_chan=64)
    # sess.placefield.pf1d.plot_with_phase(subplots=(7, 10))

    pf1d = sess.placefield.pf1d
    cells = [28, 31, 58, 61]
    ratemaps = pf1d.no_thresh["ratemaps"]
    phases = pf1d.no_thresh["phases"]
    position = pf1d.no_thresh["pos"]
    bin_cntr = pf1d.bin[:-1] + np.diff(pf1d.bin).mean() / 2

    for i, cell in enumerate(cells):
        ax = plt.subplot(gs[i, 0])
        ax.fill_between(bin_cntr, 0, ratemaps[cell], color="k", alpha=0.1)
        ax.plot(bin_cntr, ratemaps[cell], color="k", alpha=0.4)
        ax.set_ylabel("Firing rate")
        ax.spines["right"].set_visible(True)
        axphase = ax.twinx()
        axphase.scatter(position[cell], phases[cell], c="k", s=0.5)
        axphase.set_ylabel(r"$\theta$ phase")

    ax.set_xlabel("Positon (cm)")
# endregion

#%% Detect run laps scratchpad
# region
figure = Fig()
fig, gs = figure.draw(num=1, grid=(3, 1), size=(20, 7))
sessions = subjects.sd([3])
for sub, sess in enumerate(sessions):
    position = sess.tracks["maze1"]
    x = position.linear
    speed = gaussian_filter1d(position.speed, sigma=50)
    high_speed = threshPeriods(
        speed, lowthresh=10, highthresh=10, minDistance=80, minDuration=200
    )
    run_dir = np.diff(x)
    run_dir = np.where(run_dir > 0, 1, -1)

    val = []
    for epoch in high_speed:
        displacement = x[epoch[1]] - x[epoch[0]]
        distance = np.abs(np.diff(x[epoch[0] : epoch[1]])).sum()

        if np.abs(displacement) > 50:
            if displacement < 0:
                val.append(-1)

            elif displacement > 0:
                val.append(1)

        else:
            val.append(0)

    val = np.asarray(val)
    high_speed = np.delete(high_speed, np.where(val == 0)[0], axis=0)
    val = np.delete(val, np.where(val == 0)[0])

    forward = high_speed[val == 1, :]
    backward = high_speed[val == -1, :]

    ax = plt.subplot(gs[0])
    ax.plot(x)
    for epoch in forward:
        ax.axvspan(
            epoch[0], epoch[1], ymax=np.diff(epoch), facecolor="green", alpha=0.3
        )
    for epoch in backward:
        ax.axvspan(epoch[0], epoch[1], ymax=np.diff(epoch), facecolor="blue", alpha=0.3)

    ax = plt.subplot(gs[1], sharex=ax)
    # ax.fill_between(
    #     np.arange(len(speed)), 0, 50, where=speed > 10, facecolor="green", alpha=0.5
    # )
    ax.plot(speed, color="gray")
    for epoch in high_speed:
        ax.axvspan(epoch[0], epoch[1], ymax=np.diff(epoch), facecolor="gray", alpha=0.3)

    ax = plt.subplot(gs[2], sharex=ax)
    ax.plot(np.diff(x))

# endregion

#%% write to .clu file in specific order
# region
sessions = subjects.Sd().ratSday3
for sess in sessions:
    sess.placefield.pf1d.compute("maze1", run_dir="forward")
    ratemaps = sess.placefield.pf1d.no_thresh["ratemaps"]
    cell_order = np.argsort(np.argmax(np.asarray(ratemaps), axis=1))
    spikes = sess.spikes.pyr
    new_order_spks = [spikes[_] for _ in cell_order[::-1]]
    sess.spikes.export2neuroscope(new_order_spks)

# endregion
#%% Bayesian estimation in 1d linear type track
# region
sessions = subjects.Sd().ratSday3
for sess in sessions:
    sess.placefield.pf1d.compute("maze1", grid_bin=5, speed_thresh=10)
    sess.decode.bayes1d.estimate_behavior(
        sess.placefield.pf1d, binsize=0.25, speed_thresh=True
    )

    # plt.plot(gaussian_filter1d(sess.decode.bayes1d.decodedPos, sigma=1))
    # plt.plot(sess.decode.bayes1d.actualpos, sess.decode.bayes1d.decodedPos, ".")

# endregion
