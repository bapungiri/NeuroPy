#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.ndimage import gaussian_filter1d, gaussian_filter
import subjects
from plotUtil import Fig
from mathutil import threshPeriods
import seaborn as sns


#%% Ripple rate and reactivation (Explained variance)
# region

data_sd, data_rs, data_nsd = [], [], []
sessions = subjects.Sd() + subjects.Nsd()
for sub, sess in enumerate(sessions):
    post = sess.epochs.post
    rpls = sess.ripple.events.start
    tag = sess.recinfo.animal.tag

    if tag == "sd":
        # -------sd period-----------
        sd = np.linspace(post[0], post[0] + 5 * 3600, 6)
        rpls_sd = np.histogram(rpls, bins=sd)[0] / 3600

        window = [f"ZT-{_}" for _ in range(len(sd) - 1)]
        data_sd.append(pd.DataFrame({"hour": window, "ripple": rpls_sd}))

        # -------- recovery sleep -------
        states = sess.brainstates.states
        recslp_nrem = states.loc[
            (states.start > post[0] + 5 * 3600)
            & (states.name == "nrem")
            & (states.duration > 240),
            ["start", "end", "duration"],
        ]

        nrem_dur = np.asarray(recslp_nrem.duration)
        nrem_bins = recslp_nrem.to_numpy()[:, :2].flatten()
        rpls_nrem = np.histogram(rpls, bins=nrem_bins)[0][::2] / nrem_dur

        window = [nrem_ind + 1 for nrem_ind in range(len(nrem_dur))]
        data_rs.append(
            pd.DataFrame(
                {
                    "nrem": window,
                    "ripple": rpls_nrem,
                }
            )
        )

    else:
        # -------- recovery sleep -------
        states = sess.brainstates.states
        recslp_nrem = states.loc[
            (states.start > post[0])
            & (states.name == "nrem")
            & (states.duration > 240),
            ["start", "end", "duration"],
        ]

        nrem_dur = np.asarray(recslp_nrem.duration)
        nrem_bins = recslp_nrem.to_numpy()[:, :2].flatten()
        rpls_nrem = np.histogram(rpls, bins=nrem_bins)[0][::2] / nrem_dur

        window = [nrem_ind + 1 for nrem_ind in range(len(nrem_dur))]
        data_nsd.append(
            pd.DataFrame(
                {
                    "nrem": window,
                    "ripple": rpls_nrem,
                }
            )
        )


density_sd = pd.concat(data_sd)
density_rs = pd.concat(data_rs)
density_nsd = pd.concat(data_nsd)


figure = Fig()
fig, gs = figure.draw(num=1, grid=[2, 1], hspace=0.2)

gs_ = figure.subplot2grid(gs[0], grid=(2, 4))
axripple = plt.subplot(gs_[0, 0])
sns.barplot(
    x="hour", y="ripple", data=density_sd, ci="sd", ax=axripple, color="#d93a3a"
)

axripple.set_ylabel("Ripples/s")
axripple.tick_params(axis="x", labelrotation=45)
figure.panel_label(axripple, "a")

axrs = plt.subplot(gs_[0, 1:], sharey=axripple)
sns.barplot(x="nrem", y="ripple", data=density_rs, ci="sd", ax=axrs, color="#69c")
axrs.set_ylabel("")

axnsd = plt.subplot(gs_[1, :], sharey=axripple)
sns.barplot(x="nrem", y="ripple", data=density_nsd, ci="sd", ax=axnsd, color="#69c")
axnsd.set_ylabel("Ripples/s")

sessions = (
    subjects.Nsd().ratSday2
    + subjects.Sd().ratSday3
    + subjects.Nsd().ratNday2
    + subjects.Sd().ratNday1
)
# sessions = subjects.Sd().ratSday3
gs2 = figure.subplot2grid(gs[1], grid=(2, 2))

for sub, sess in enumerate(sessions):

    pre = sess.epochs.pre

    try:
        maze = sess.epochs.maze
    except:
        maze = sess.epochs.maze1
    # maze2 = sess.epochs.maze2

    post = sess.epochs.post
    # maze2 = sess.epochs.maze2
    # --- break region into periods --------
    bin1 = sess.utils.getinterval(pre, 2)
    bin2 = sess.utils.getinterval(post, 4)
    bins = bin1 + bin2
    sess.spikes.stability.firingRate(periods=bins)

    control = pre
    template = maze
    match = [post[0], post[1]]

    sess.replay.expvar.compute(
        template=template,
        match=match,
        control=control,
        slideby=300,
        cross_shanks=True,
    )
    print(sess.replay.expvar.npairs)

    axstate = figure.subplot2grid(gs2[sub], grid=(4, 1))

    ax1 = fig.add_subplot(axstate[1:])
    sess.replay.expvar.plot(ax=ax1, tstart=post[0])
    ax1.set_xlim(left=0)
    ax1.tick_params(width=2)

    if sub == 3:
        ax1.set_ylim([0, 0.17])
    # ax1.spines["right"].set_visible("False")
    # ax1.spines["top"].set_visible("False")

    axhypno = fig.add_subplot(axstate[0], sharex=ax1)
    sess.brainstates.hypnogram(ax=axhypno, tstart=post[0], unit="h")
    # panel_label(axhypno, "a")
    # ax1.set_ylim([0, 11])


figure.savefig("nimh1", __file__)

# endregion

#%% Explained variance and Plot 1D place fields maze1 vs maze2
# region
figure = Fig()
fig, gs = figure.draw(num=1, grid=(4, 1), hspace=0.3)

gs1 = figure.subplot2grid(gs[1], grid=(1, 2))
sessions = subjects.Nsd().ratSday2 + subjects.Sd().ratSday3

for sub, sess in enumerate(sessions):

    pre = sess.epochs.pre

    try:
        maze = sess.epochs.maze
    except:
        maze = sess.epochs.maze1
    maze2 = sess.epochs.maze2

    post = sess.epochs.post

    # maze2 = sess.epochs.maze2
    # --- break region into periods --------
    bin1 = sess.utils.getinterval(pre, 2)
    bin2 = sess.utils.getinterval(post, 4)
    bins = bin1 + bin2
    sess.spikes.stability.firingRate(periods=bins)

    control = pre
    template = maze
    match = [post[0], maze2[1]]

    sess.replay.expvar.compute(
        template=template,
        match=match,
        control=control,
        slideby=300,
        cross_shanks=True,
    )
    print(sess.replay.expvar.npairs)

    axstate = figure.subplot2grid(gs1[sub], grid=(4, 1))

    ax1 = fig.add_subplot(axstate[1:])
    sess.replay.expvar.plot(ax=ax1, tstart=post[0])
    ax1.set_xlim(left=0)
    ax1.tick_params(width=2)

    if sub == 3:
        ax1.set_ylim([0, 0.17])
    # ax1.spines["right"].set_visible("False")
    # ax1.spines["top"].set_visible("False")

    axhypno = fig.add_subplot(axstate[0], sharex=ax1)
    sess.brainstates.hypnogram(ax=axhypno, tstart=post[0], unit="h")
    # panel_label(axhypno, "a")
    # ax1.set_ylim([0, 11])

gs2 = figure.subplot2grid(gs[2:], grid=(1, 2), wspace=0.3)
sessions = subjects.Sd().ratSday3
for sub, sess in enumerate(sessions):
    # period = sess.epochs.maze1
    sess.placefield.pf1d.compute(track_name="maze1", run_dir="forward")

    ratemaps = np.asarray(sess.placefield.pf1d.no_thresh["ratemaps"])
    peak_frate = np.max(ratemaps, axis=1)
    good_cells = np.where(peak_frate > 1.5)[0]

    good_ratemaps = ratemaps[good_cells, :]
    cell_order = np.argsort(np.argmax(good_ratemaps, axis=1))
    good_cells = good_cells[cell_order]

    gs3 = figure.subplot2grid(gs2[0], grid=(1, 2))

    ax = plt.subplot(gs3[0])
    sess.placefield.pf1d.plot(
        ax=ax, speed_thresh=True, normalize=True, sortby=good_cells
    )
    # sess.placefield.pf1d.plot_raw(speed_thresh=True, subplots=None)
    ax.set_title("Maze1")
    ax = plt.subplot(gs3[1])
    sess.placefield.pf1d.compute(track_name="maze2", run_dir="forward")
    sess.placefield.pf1d.plot(
        ax=ax,
        speed_thresh=True,
        normalize=True,
        sortby=good_cells,
    )
    ax.set_title("Maze2")
    # sess.placefield.pf1d.plot_raw(speed_thresh=True, subplots=None)

    gs4 = figure.subplot2grid(gs2[1], grid=(1, 2))
    sess.placefield.pf1d.compute(track_name="maze1", run_dir="backward")

    ratemaps = np.asarray(sess.placefield.pf1d.no_thresh["ratemaps"])
    peak_frate = np.max(ratemaps, axis=1)
    good_cells = np.where(peak_frate > 1.5)[0]

    good_ratemaps = ratemaps[good_cells, :]
    cell_order = np.argsort(np.argmax(good_ratemaps, axis=1))
    good_cells = good_cells[cell_order]

    ax = plt.subplot(gs4[0])
    sess.placefield.pf1d.plot(
        ax=ax, speed_thresh=True, normalize=True, sortby=good_cells
    )
    # sess.placefield.pf1d.plot_raw(speed_thresh=True, subplots=None)
    ax.set_title("Maze1")
    ax = plt.subplot(gs4[1])
    sess.placefield.pf1d.compute(track_name="maze2", run_dir="backward")
    sess.placefield.pf1d.plot(
        ax=ax,
        speed_thresh=True,
        normalize=True,
        sortby=good_cells,
    )
    ax.set_title("Maze2")
    # sess.placefield.pf1d.plot_raw(speed_thresh=True, subplots=None)
figure.savefig("nimh2", __file__)
# endregion
