#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter1d, gaussian_filter
import subjects
from plotUtil import Fig
from mathutil import threshPeriods
import time
import seaborn as sns

#%% 1D place field in openfield arena
# region
sessions = subjects.Of()
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
fig, gs = figure.draw(num=1, grid=(2, 2))
sessions = subjects.Sd().ratSday3
for sub, sess in enumerate(sessions):
    # period = sess.epochs.maze1
    ax = plt.subplot(gs[2])
    sess.placefield.pf1d.compute(track_name="maze1", run_dir="forward")

    ratemaps = np.asarray(sess.placefield.pf1d.ratemaps["ratemaps"])
    peak_frate = np.max(ratemaps, axis=1)
    good_cells = np.where(peak_frate > 1.5)[0]

    good_ratemaps = ratemaps[good_cells, :]
    cell_order = np.argsort(np.argmax(good_ratemaps, axis=1))
    good_cells = good_cells[cell_order]

    sess.placefield.pf1d.plot(ax=ax, normalize=True, sortby=good_cells)
    # sess.placefield.pf1d.plot_raw(speed_thresh=True, subplots=None)
    ax.set_title("Maze1")
    ax = plt.subplot(gs[3])
    sess.placefield.pf1d.compute(track_name="maze2", run_dir="forward")
    sess.placefield.pf1d.plot(
        ax=ax,
        normalize=True,
        sortby=good_cells,
    )
    ax.set_title("Maze2")
    # sess.placefield.pf1d.plot_raw(speed_thresh=True, subplots=None)

# endregion

#%% Theta-precession in linear type tracks
# region
figure = Fig()
fig, gs = figure.draw(num=1, grid=(4, 2), size=(5, 12))
sessions = subjects.Sd().ratSday3
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
sessions = subjects.Sd().ratNday1
for sess in sessions:
    sess.placefield.pf1d.compute("maze", grid_bin=8, speed_thresh=5)
    sess.decode.bayes1d.binsize = 0.25
    sess.decode.bayes1d.estimate_behavior(speed_thresh=True, smooth=2)

    # plt.plot(gaussian_filter1d(sess.decode.bayes1d.decodedPos, sigma=1))
    # plt.plot(sess.decode.bayes1d.actualpos, sess.decode.bayes1d.decodedPos, ".")

# endregion

#%% Decode pbe events
# region
figure = Fig()
sessions = subjects.Nsd().ratSday2
for sess in sessions:
    sd_period = sess.epochs["post"]
    maze = sess.epochs.post
    rpls = sess.pbe.events
    rpls = rpls[
        (rpls.start > maze[0] + 5 * 3600)
        & (rpls.start < maze[1] + 6 * 3600)
        & (rpls.duration > 0.15)
        & (rpls.duration < 0.45)
    ]
    sess.placefield.pf1d.compute("maze1", grid_bin=8, run_dir="forward")
    sess.decode.bayes1d.events = rpls
    sess.decode.bayes1d.binsize = 0.02
    sess.decode.bayes1d.n_jobs = 12

    sess.decode.bayes1d.decode_events()
    score = sess.decode.bayes1d.score
    slope = sess.decode.bayes1d.slope
    posterior = sess.decode.bayes1d.posterior

    st = time.time()
    sess.decode.bayes1d.decode_shuffle(n_iter=100, kind="column")
    shuff_score = sess.decode.bayes1d.shuffle_score
    print(time.time() - st)

    pval = sess.decode.bayes1d.p_val_events

    bins = np.arange(0, 0.7, 0.01)
    hist_evt = np.histogram(score, bins=bins)[0]
    hist_shuffle = np.histogram(shuff_score, bins=bins)[0]

    good_evt_ind = np.where(pval < 0.05)[0]
    top_10 = np.argsort(score[good_evt_ind])[::-1][:100]
    good_evt = [posterior[_] for _ in good_evt_ind[top_10]]
    score_top = score[good_evt_ind[top_10]]
    # slope_top = slope[good_evt_ind[top_10]]

    # postive = np.where(slope_top > 0.2)[0]
    # score_top = score_top[postive]
    # slope_top = slope_top[postive]
    fig, gs = figure.draw(num=1, grid=(8, 12), hspace=0.3)
    for i in range(100):

        ax = plt.subplot(gs[i])
        ax.pcolormesh(good_evt[i], cmap="hot", vmin=0, vmax=0.4)
        ax.set_title(f"{np.round(score_top[i],2)}")
        ax.axis("off")

    # ind_ = np.arange(0, len(good_evt), 200)
    # n_fig = int(len(good_evt) / 200)
    # for fig_wind in range(n_fig):
    #     p = ind_[fig_wind]
    #     fig, gs = figure.draw(num=fig_wind, grid=(10, 20))
    #     for ind, i in enumerate(range(p, p + 200)):
    #         ax = plt.subplot(gs[ind])
    #         posterior = good_evt[i]  # / np.max(b[i])
    #         ax.pcolormesh(posterior, cmap="hot")
    # ax.xticks([])

    # fig, gs = figure.draw(num=2, grid=(1, 1))

    # ax = plt.subplot(gs[0])
    # ax.bar(bins[:-1], hist_shuffle / np.sum(hist_shuffle), width=0.01, color="#aca5a5")
    # ax.axvline(thresh, color="r")
    # ax.set_xlabel("Replay score")
    # ax.set_ylabel("Density")


# endregion

#%% Significant replay events across time e.g, POST
# region

figure = Fig()
fig, gs = figure.draw(num=1, grid=(4, 2))
sessions = (
    subjects.Sd().ratNday1
    + subjects.Nsd().ratNday2
    + subjects.Sd().ratSday3
    + subjects.Nsd().ratSday2
)
for sub, sess in enumerate(sessions):
    sd_period = sess.epochs["post"]
    track_name = sess.tracks.names[0]
    maze = sess.epochs[track_name]
    # maze2 = sess.epochs.maze2
    post = sess.epochs.post

    period = [post[0], post[1]]
    period_bins = np.arange(period[0], period[1], 900)
    events = sess.pbe.events
    events = events[
        (events.start > period[0])
        & (events.start < period[1])
        & (events.duration > 0.1)
        & (events.duration < 0.5)
    ]

    binsize = 0.02
    sess.decode.bayes1d.events = events
    sess.decode.bayes1d.binsize = 0.02
    sess.decode.bayes1d.n_jobs = 12

    score, slope, shuffle_score = [], [], []
    replay_hist = pd.DataFrame()
    bincntr = (period_bins[:-1] - period[0] + 450) / 3600
    for run_dir in ["forward", "backward"]:
        sess.placefield.pf1d.compute(track_name, grid_bin=8, run_dir=run_dir)
        sess.decode.bayes1d.decode_events()
        score_ = sess.decode.bayes1d.score
        slope_ = np.abs(sess.decode.bayes1d.slope)
        sess.decode.bayes1d.decode_shuffle(n_iter=200, kind="column")
        pval = sess.decode.bayes1d.p_val_events

        good_evt_ind = np.where(pval < 0.05)[0]
        # good_evt_ind = np.where(score_ > 0.2)[0]
        slope_good = slope_[good_evt_ind]
        hist_ = np.histogram(events.iloc[good_evt_ind].start, bins=period_bins)[0]
        mean_slope = stats.binned_statistic(
            events.iloc[good_evt_ind].start, slope_good, bins=period_bins
        )[0]
        replay_hist = replay_hist.append(
            pd.DataFrame(
                {
                    "time": bincntr,
                    "number": hist_,
                    "direction": run_dir,
                    "slope": mean_slope,
                }
            )
        )

    gs_ = figure.subplot2grid(gs[sub], grid=(4, 1))
    ax = plt.subplot(gs_[1:])
    sns.lineplot(
        data=replay_hist,
        x="time",
        y="number",
        hue="direction",
        ax=ax,
        palette=["#5d538d", "#f95939"],
    )
    ax.set_xlim([0, np.diff(period) / 3600])
    plt.setp(ax.get_xticklabels(), visible=False)
    ax.set_xlabel("")

    # axvel = plt.subplot(gs_[3:])
    # sns.lineplot(
    #     data=replay_hist.groupby("time").mean(),
    #     x="time",
    #     y="slope",
    #     ax=axvel,
    #     color="k",
    # )

    axhypno = plt.subplot(gs_[0], sharex=ax)
    sess.brainstates.hypnogram(ax=axhypno, tstart=period[0], unit="h")

figure.savefig("replay_across_time2", __file__)

# endregion
