import matplotlib.pyplot as plt
import numpy as np
from numpy.core.fromnumeric import sort
import pandas as pd
import scipy.stats as stats
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter1d, gaussian_filter
import time
from callfunc import processData


basePath = [
    # "/data/Clustering/SleepDeprivation/RatJ/Day1/",
    # "/data/Clustering/SleepDeprivation/RatK/Day1/",
    # "/data/Clustering/SleepDeprivation/RatN/Day1/",
    # "/data/Clustering/SleepDeprivation/RatJ/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day2/",
    # "/data/Clustering/SleepDeprivation/RatN/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day4/"
    "/data/Clustering/SleepDeprivation/RatN/Day4/",
]


sessions = [processData(_) for _ in basePath]

plt.clf()
fig = plt.figure(1, figsize=(8.5, 11))
gs = gridspec.GridSpec(6, 7, figure=fig)
fig.subplots_adjust(hspace=0.5, wspace=0.5)
fig.suptitle("Bayesian decoding in Open Field")
titlesize = 8
panel_label = lambda ax, label: ax.text(
    x=-0.08,
    y=1.15,
    s=label,
    transform=ax.transAxes,
    fontsize=12,
    fontweight="bold",
    va="top",
    ha="right",
)


#%% Plotting all replay trajectory
# region

for sub, sess in enumerate(sessions):
    sess.trange = np.array([])
    speed = sess.position.speed
    posx = sess.position.x[1:]
    posy = sess.position.y[1:]
    pos_time = sess.position.t[1:]
    maze = sess.epochs.maze
    post = sess.epochs.post
    pbe = sess.pbe.events
    maze_pbe = pbe[(pbe.start > maze[0]) & (pbe.start < maze[1])].reset_index()

    pbe_bin = maze_pbe[["start", "end"]].to_numpy().reshape(len(maze_pbe) * 2)
    pbe_speed = stats.binned_statistic(pos_time, speed, bins=pbe_bin)[0][::2]
    pbe_posx = stats.binned_statistic(pos_time, posx, bins=pbe_bin)[0][::2]
    pbe_posy = stats.binned_statistic(pos_time, posy, bins=pbe_bin)[0][::2]

    immobile_pbe = maze_pbe.drop((np.where(pbe_speed > 5)[0])).reset_index()
    pbe_posx = np.delete(pbe_posx, np.where(pbe_speed > 5)[0])
    pbe_posy = np.delete(pbe_posy, np.where(pbe_speed > 5)[0])

    sess.decode.bayes2d.fit()
    grid = sess.decode.bayes2d.grid
    decodedPos, posterior = sess.decode.bayes2d.decode(immobile_pbe, binsize=0.02)
    jumpDist = [np.sqrt((np.diff(arr, axis=1) ** 2).sum(axis=0)) for arr in decodedPos]

    plt.clf()
    ax = fig.add_subplot(gs[:2, 5:7])
    hist_jump, edges = np.histogram(np.concatenate(jumpDist), bins=30)
    ax.plot(edges[:-1], hist_jump)
    ax.set_yscale("log")
    ax.set_ylabel("Counts")
    ax.set_xlabel("Jump distance")

    inner = gridspec.GridSpecFromSubplotSpec(
        4, 4, subplot_spec=gs[:6, :4], hspace=0.2, wspace=0.1
    )

    for i, ind in enumerate(np.random.randint(0, len(immobile_pbe), 16)):

        ax = fig.add_subplot(inner[i])
        ax.clear()
        pbe_posterior = posterior[ind].sum(axis=1).reshape(49, 49)
        trajectory = decodedPos[ind]
        ax.pcolormesh(grid[0], grid[1], (pbe_posterior), cmap="hot", vmax=0.4)
        ax.plot(
            gaussian_filter1d(trajectory[0], sigma=1),
            gaussian_filter1d(trajectory[1], sigma=1),
            "#25cdd0",
        )
        ax.plot(
            posx[
                np.where(
                    (pos_time > immobile_pbe.start[ind] - 10)
                    & (pos_time < immobile_pbe.start[ind])
                )[0]
            ],
            posy[
                np.where(
                    (pos_time > immobile_pbe.start[ind] - 10)
                    & (pos_time < immobile_pbe.start[ind])
                )[0]
            ],
            "b",
        )
        ax.plot(
            posx[
                np.where(
                    (pos_time > immobile_pbe.end[ind])
                    & (pos_time < immobile_pbe.end[ind] + 10)
                )[0]
            ],
            posy[
                np.where(
                    (pos_time > immobile_pbe.end[ind])
                    & (pos_time < immobile_pbe.end[ind] + 10)
                )[0]
            ],
            "r",
        )
        ax.plot(pbe_posx[ind], pbe_posy[ind], "*", color="#fe938b")
        ax.axis("off")
        ax.set_title(
            f"{round(immobile_pbe.duration[ind],2)}, evt= {ind}", fontsize=titlesize
        )

    avg_traj_speed = np.array([np.mean(_) for _ in jumpDist])
    sort_ind = np.argsort(avg_traj_speed)
    jumpDist_srtd = [jumpDist[_] for _ in sort_ind]
    mean_spd = np.array([np.mean(jumpDist[_]) for _ in sort_ind])
    std_spd = np.array([np.std(jumpDist[_]) for _ in sort_ind])

    ax = fig.add_subplot(gs[2:4, 5:7])
    ax.errorbar(np.arange(len(mean_spd)), mean_spd, yerr=std_spd, ecolor="gray")
    ax.set_xlabel("Events")
    ax.set_ylabel("Mean Trajectory \n speed")

    dist = np.concatenate(
        [
            np.sqrt(((arr[:, 1:] - arr[:, 0][:, np.newaxis]) ** 2).sum(axis=0))
            for arr in decodedPos
        ]
    )
    step = np.concatenate([np.arange(1, arr.shape[1]) for arr in decodedPos])

    step_bin = np.arange(1, 26)
    mean_val = stats.binned_statistic(step, dist, bins=step_bin)[0]
    std_val = stats.binned_statistic(step, dist, bins=step_bin, statistic="std")[0]
    count_val = stats.binned_statistic(step, dist, bins=step_bin, statistic="count")[0]

    slope = stats.linregress(np.log10(step_bin[:-1]), np.log10(mean_val))[0]

    ax = fig.add_subplot(gs[4:6, 5:7])
    ax.errorbar(step_bin[:-1], mean_val, yerr=std_val / np.sqrt(count_val))
    ax.text(10, 60, f"alpha={round(slope,2)}")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Avg. distance of \n encoded points (cm)")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_ylim([20, 150])
    # ax.set_xlabel("Jump distance (cm)")
    # ax.set_ylabel("Counts")


# endregion

