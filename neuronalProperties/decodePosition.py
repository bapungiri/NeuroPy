#%% Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter1d, gaussian_filter
import time
import subjects

#%% Bayesian decoding in open field
# region
sessions = subjects.Of().ratNday4
for sub, sess in enumerate(sessions):

    maze = sess.epochs.maze
    track = sess.tracks.data["maze"]
    sess.placefield.pf2d.compute(period=maze)
    ratemaps = sess.placefield.pf2d.ratemaps
    xgrid = sess.placefield.pf2d.xgrid
    ygrid = sess.placefield.pf2d.ygrid
    sess.decode.bayes2d.estimate_behavior()
    dec_pos = sess.decode.bayes2d.decodedPos
    sess.decode.bayes2d.events = sess.pbe.events
    # sess.decode.bayes2d.plot()


# endregion

#%% Decoding population burst events during MAZE of open field env
# region
for sub, sess in enumerate(sessions):
    sess.trange = np.array([])
    maze = sess.epochs.maze
    pbe = sess.pbe.events
    maze_pbe = pbe[(pbe.start > maze[0]) & (pbe.start < maze[1])]
    sess.decode.bayes2d.fit()
    decodedPos = sess.decode.bayes2d.decode(maze_pbe, binsize=0.02)
    jumpDist = [np.sqrt((np.diff(arr, axis=1) ** 2).sum(axis=0)) for arr in decodedPos]
    avg_traj_speed = np.array([np.mean(_) for _ in jumpDist])
    sort_ind = np.argsort(avg_traj_speed)
    jumpDist_srtd = [jumpDist[_] for _ in sort_ind]
    mean_spd = np.array([np.mean(jumpDist[_]) for _ in sort_ind])
    std_spd = np.array([np.std(jumpDist[_]) for _ in sort_ind])
    plt.errorbar(np.arange(len(mean_spd)), mean_spd, yerr=std_spd)

    dist = np.concatenate(
        [
            np.sqrt(((arr[:, 1:] - arr[:, 0][:, np.newaxis]) ** 2).sum(axis=0))
            for arr in decodedPos
        ]
    )
    step = np.concatenate([np.arange(1, arr.shape[1]) for arr in decodedPos])

    step_bin = np.arange(1, 11)
    mean_val = stats.binned_statistic(step, dist, bins=step_bin)[0]

    slope = stats.linregress(np.log10(step_bin[:-1]), np.log10(mean_val))[0]

# endregion

#%% Comparing jump distance between L-shaped vs open field
# region
plt.clf()
fig = plt.figure(1, figsize=(10, 15))
gs = gridspec.GridSpec(1, 1, figure=fig)
fig.subplots_adjust(hspace=0.3)
for sub, sess in enumerate(sessions):
    sess.trange = np.array([])
    maze = sess.epochs.maze
    pbe = sess.pbe.events
    maze_pbe = pbe[(pbe.start > maze[0]) & (pbe.start < maze[1])]
    sess.decode.bayes2d.fit()
    decodedPos = sess.decode.bayes2d.decode(maze_pbe, binsize=0.02)
    jumpDist = [np.sqrt((np.diff(arr, axis=1) ** 2).sum(axis=0)) for arr in decodedPos]
    avg_traj_speed = np.array([np.mean(_) for _ in jumpDist])
    sort_ind = np.argsort(avg_traj_speed)
    hist_jumpDist = np.histogram(np.concatenate(jumpDist), bins=np.arange(0, 300, 10))[
        0
    ]

    ax = fig.add_subplot(gs[0])
    ax.plot(np.arange(0, 300, 10)[:-1], hist_jumpDist)
    ax.set_xlabel("Jump distance (cm)")
    ax.set_ylabel("Counts")


# endregion

#%% Plotting all replay trajectory
# region
plt.clf()
fig = plt.figure(1, figsize=(10, 15))
gs = gridspec.GridSpec(4, 4, figure=fig)
fig.subplots_adjust(hspace=0.1)

for sub, sess in enumerate(sessions):
    sess.trange = np.array([])
    speed = sess.position.speed
    posx = sess.position.x
    posy = sess.position.y
    pos_time = sess.position.t[1:]
    maze = sess.epochs.maze
    pbe = sess.pbe.events
    maze_pbe = pbe[(pbe.start > maze[0]) & (pbe.start < maze[1])].reset_index()

    pbe_bin = maze_pbe[["start", "end"]].to_numpy().reshape(len(maze_pbe) * 2)
    pbe_speed = stats.binned_statistic(pos_time, speed, bins=pbe_bin)[0][::2]
    pbe_posx = stats.binned_statistic(pos_time, posx[1:], bins=pbe_bin)[0][::2]
    pbe_posy = stats.binned_statistic(pos_time, posy[1:], bins=pbe_bin)[0][::2]

    immobile_pbe = maze_pbe.drop((np.where(pbe_speed > 5)[0])).reset_index()
    pbe_posx = np.delete(pbe_posx, np.where(pbe_speed > 5)[0])
    pbe_posy = np.delete(pbe_posy, np.where(pbe_speed > 5)[0])

    sess.decode.bayes2d.fit()
    grid = sess.decode.bayes2d.grid
    decodedPos, posterior = sess.decode.bayes2d.decode(immobile_pbe, binsize=0.02)
    jumpDist = [np.sqrt((np.diff(arr, axis=1) ** 2).sum(axis=0)) for arr in decodedPos]

    for i, ind in enumerate(np.random.randint(0, len(immobile_pbe), 16)):

        ax = fig.add_subplot(gs[i])
        ax.clear()
        pbe_posterior = posterior[ind].sum(axis=1).reshape(49, 49)
        trajectory = decodedPos[ind]
        ax.pcolormesh(grid[0], grid[1], (pbe_posterior), cmap="hot", vmax=0.4)
        ax.plot(
            gaussian_filter1d(trajectory[0], sigma=1),
            gaussian_filter1d(trajectory[1], sigma=1),
            "b",
        )
        ax.plot(pbe_posx[ind], pbe_posy[ind], "*")
        ax.axis("off")
        ax.set_title(f"{round(immobile_pbe.duration[ind],2)}")
    # ax.set_xlabel("Jump distance (cm)")
    # ax.set_ylabel("Counts")


# endregion
