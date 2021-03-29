#%% Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter1d
import subjects
from plotUtil import Fig
import pingouin as pg

#%% Explained variance PRE-MAZE-POST
# region
figure = Fig()
fig, gs = figure.draw(num=1, grid=(2, 2))
# sessions = (
#     subjects.Nsd().ratSday2
#     + subjects.Sd().ratSday3
#     + subjects.Nsd().ratNday2
#     + subjects.Sd().ratNday1
# )
sessions = subjects.Tn().ratSday5

for sub, sess in enumerate(sessions):

    pre = sess.epochs.pre

    try:
        maze = sess.epochs.maze
    except:
        maze = sess.epochs.maze2
    # maze2 = sess.epochs.maze2

    post = sess.epochs.post
    # maze2 = sess.epochs.maze2
    # --- break region into periods --------
    bin1 = sess.utils.getinterval(pre, 2)
    bin2 = sess.utils.getinterval(post, 4)
    bins = bin1 + bin2
    # bins = [
    #     pre,
    #     # maze1,
    #     # [post[0] + 4 * 3600, post[0] + 5 * 3600],
    #     # [post[0] + 5 * 3600, post[0] + 10 * 4600],
    #     # post,
    #     # [post[0], maze2[1]]
    #     # [post[0] + 5 * 3600, post[0] + 8 * 3600],
    # ]

    sess.spikes.stability.firingRate(periods=bins)
    # sess.spikes.stability.refPeriodViolation()
    # violations = sess.spikes.stability.violations

    control = pre
    template = maze  # [post[0] + 4 * 3600, post[0] + 5 * 3600]
    match = post  # [post[0] + 5 * 3600, post[1]]

    sess.replay.expvar.compute(
        template=template,
        match=match,
        control=control,
        slideby=300,
        cross_shanks=True,
    )
    print(sess.replay.expvar.npairs)

    axstate = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[sub], hspace=0.2)

    ax1 = fig.add_subplot(axstate[1:])
    sess.replay.expvar.plot(ax=ax1, tstart=post[0])
    ax1.set_xlim(left=0)
    ax1.tick_params(width=2)
    # if sub == 3:
    #     ax1.set_ylim([0, 0.17])
    # ax1.spines["right"].set_visible("False")
    # ax1.spines["top"].set_visible("False")

    # axhypno = fig.add_subplot(axstate[0], sharex=ax1)
    # sess.brainstates.hypnogram(ax=axhypno, tstart=post[0], unit="h")
    # panel_label(axhypno, "a")
    # ax1.set_ylim([0, 11])


# figure.savefig("EV_sessions", __file__)
# endregion


#%% Explained variance during recovery sleep while controlling for MAZE correlations
# region
"""Only found stable units for 3 sessions
"""
figure = Fig()
fig, gs = figure.draw(num=1, grid=(2, 1))
sessions = subjects.sd([3])
for sub, sess in enumerate(sessions):

    maze = sess.epochs.maze1
    post = sess.epochs.post
    pre = sess.epochs.pre

    for i, hour in enumerate(range(5, 10)):
        bins = [
            # [pre[0] + 2 * 3600, pre[0] + 3 * 3600],
            # [maze[0], maze[1]],
            [post[0], post[0] + 3600],
            [post[0] + 4 * 3600, post[0] + 5 * 3600],
            [post[0] + hour * 3600, post[0] + (hour + 1) * 3600],
        ]
        # sess.spikes.stability.refPeriodViolation()
        # violations = sess.spikes.stability.violations
        sess.spikes.stability.firingRate(bins=bins)
        sess.replay.expvar.compute(
            template=bins[1],
            match=bins[2],
            control=bins[0],
        )

        axstate = gridspec.GridSpecFromSubplotSpec(
            4, 1, subplot_spec=gs[sub, i], hspace=0.2
        )

        ax1 = fig.add_subplot(axstate[1:])
        sess.replay.expvar.plot(ax=ax1, tstart=post[0])
        if sub == 0:
            ax1.set_ylim([0, 0.55])
        else:
            ax1.set_ylim([0, 0.4])

        if i > 0:
            ax1.spines["left"].set_visible(False)
            ax1.set_yticks([])
            ax1.set_yticklabels([])
            ax1.set_ylabel("")
            ax1.legend("")
            ax1.set_title("")

        axhypno = fig.add_subplot(axstate[0], sharex=ax1)
        sess.brainstates.hypnogram(ax1=axhypno, tstart=post[0], unit="h")
        # axhypno.set_title(sess.sessinfo.session.sessionName)
    # panel_label(axhypno, "a")
    # ax1.set_ylim([0, 0.3])


# endregion

#%% Explained variance in recovery sleep
# region
figure = Fig()
fig, gs = figure.draw(num=1, grid=(6, 1))
sessions = subjects.Sd().ratSday3

for sub, sess in enumerate(sessions):

    sess.trange = np.array([])
    pre = sess.epochs.pre
    maze = sess.epochs.maze1
    post = sess.epochs.post

    template_periods = [
        [post[0] + _ * 3600, post[0] + (_ + 1) * 3600] for _ in range(5)
    ]

    for i, period in enumerate(template_periods):
        bins = [
            maze,
            period,
            [post[0] + 5 * 3600, post[0] + 10 * 4600],
        ]
        sess.spikes.stability.firingRate(bins=bins)
        sess.replay.expvar.compute(template=bins[1], match=bins[2], control=bins[0])

        ax1 = fig.add_subplot(gs[i + 1])
        sess.replay.expvar.plot(ax=ax1, tstart=post[0])
        ax1.set_xlim(left=0)
        ax1.set_ylim(top=0.35)
        h = np.array(period) / 3600 - post[0] / 3600
        ax1.axvspan(xmin=h[0], xmax=h[1], color="red", alpha=0.5)
        # ax1.spines["right"].set_visible("False")
        # ax1.spines["top"].set_visible("False")

    axstate = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[0], hspace=0.2)
    axhypno = fig.add_subplot(axstate[3], sharex=ax1)
    sess.brainstates.hypnogram(ax1=axhypno, tstart=post[0], unit="h")
    # panel_label(axhypno, "a")
    # ax1.set_ylim([0, 11])


# endregion

#%% Pairwise correlation across time each session individually
# region
figure = Fig()
fig, gs = figure.draw(num=1, grid=(3, 2))
# sessions = subjects.Sd().allsess + subjects.Nsd().allsess
sessions = subjects.Sd().ratJday1

for sub, sess in enumerate(sessions):

    pre = sess.epochs.pre
    maze1 = sess.epochs.maze
    if maze1 is None:
        maze1 = sess.epochs.maze1
    post = sess.epochs.post

    # --- break region into periods --------
    bins = sess.utils.getinterval([post[0], post[0] + 5 * 3600], 2)
    sess.spikes.stability.firingRate(periods=bins)

    sess.replay.corr.across_time_window(period=[post[0], post[0] + 5 * 3600])
    ax = plt.subplot(gs[sub])
    sess.replay.corr.plot_across_time(ax=ax, smooth=1, tstart=maze1[0])

    corr_mat = sess.replay.corr.corr


# figure.savefig("RatS_EV", __file__)
# endregion

#%% Pairwise correlation across pooled across sessions
# region
sessions = subjects.Sd().allsess + subjects.Nsd().allsess
# sessions = subjects.Sd().ratJday1
corr_sd, corr_nsd = [], []
for sub, sess in enumerate(sessions):

    pre = sess.epochs.pre
    maze1 = sess.epochs.maze
    if maze1 is None:
        maze1 = sess.epochs.maze1
    post = sess.epochs.post
    tag = sess.recinfo.animal.tag

    # --- break region into periods --------
    # bins = sess.utils.getinterval([post[0], post[0] + 5 * 3600], 2)
    # sess.spikes.stability.firingRate(periods=bins)
    spks = sess.spikes.pyr
    epochs = np.arange(post[0], post[0] + 8 * 3600, 300)
    corr = []
    for i in range(len(epochs) - 1):
        corr.append(
            sess.spikes.corr.pairwise(spikes=spks, period=[epochs[i], epochs[i + 1]])
        )
    corr = np.asarray(corr)

    if tag == "sd":
        corr_sd.append(corr)
    else:
        corr_nsd.append(corr)


def process_(corr_mat):
    corr_mat = np.hstack(corr_mat)
    corr_mask = np.ma.array(corr_mat, mask=np.isnan(corr_mat))
    corr_ = np.ma.corrcoef(corr_mask)
    np.fill_diagonal(corr_, 0)
    # corr_ = gaussian_filter(corr_, sigma=1)
    return corr_


corr_sd = process_(corr_sd)
corr_nsd = process_(corr_nsd)

time = np.linspace(0, 8, corr_sd.shape[0] + 1)

figure = Fig()
fig, gs = figure.draw(num=1, grid=(4, 3), wspace=0.3)
axsd = plt.subplot(gs[2])
im = axsd.pcolormesh(time[:-1], time[:-1], corr_sd, cmap="Spectral_r", rasterized=True)
axsd.set_xlabel("Time (h)")
axsd.set_ylabel("Time (h)")
cbar = fig.colorbar(im, ax=axsd)
cbar.outline.set_linewidth(1)


axnsd = plt.subplot(gs[5])
im = axnsd.pcolormesh(
    time[:-1], time[:-1], corr_nsd, cmap="Spectral_r", rasterized=True
)
axnsd.set_xlabel("Time (h)")
cbar = fig.colorbar(im, ax=axnsd)
cbar.outline.set_linewidth(1)
# cbar.set_label("correlation")

figure.savefig("correlation_time_window", __file__)
# endregion

#%% pooled explained variance across all sessions
# region
sessions = subjects.Sd().allsess + subjects.Nsd().allsess
# sessions = subjects.Sd().ratJday1
corr_sd, corr_nsd = [], []
slp_sd, slp_nsd = [], []
for sub, sess in enumerate(sessions):

    pre = sess.epochs.pre
    maze = sess.epochs.maze
    if maze is None:
        maze = sess.epochs.maze1
    post = sess.epochs.post
    tag = sess.recinfo.animal.tag

    # --- break region into periods --------
    # bins = sess.utils.getinterval([post[0], post[0] + 5 * 3600], 2)
    # sess.spikes.stability.firingRate(periods=bins)
    spks = sess.spikes.pyr

    corr_maze = sess.replay.corr.pairwise(spikes=spks, period=maze)[np.newaxis, :]

    epochs = np.arange(post[0], post[0] + 8 * 3600 - 900, 300)
    corr = []
    slp_frac = []
    for i in range(len(epochs)):
        corr.append(
            sess.replay.corr.pairwise(spikes=spks, period=[epochs[i], epochs[i] + 900])
        )
        slp = sess.brainstates.proportion(period=[epochs[i], epochs[i] + 900])
        if "nrem" in slp.T:
            slp_frac.append(slp.T.nrem.values[0])
        else:
            slp_frac.append(0)

    corr = np.asarray(corr)
    slp_frac = np.asarray(slp_frac)

    corr = np.append(corr_maze, corr, axis=0)

    if tag == "sd":
        corr_sd.append(corr)
        slp_sd.append(slp_frac)
    else:
        corr_nsd.append(corr)
        slp_nsd.append(slp_frac)


def process2(corr_mat):
    corr_mat = np.hstack(corr_mat)
    corr_mask = np.ma.array(corr_mat, mask=np.isnan(corr_mat))
    corr_ = np.ma.corrcoef(corr_mask)
    np.fill_diagonal(corr_, 0)
    # corr_ = gaussian_filter(corr_, sigma=1)
    return corr_


corr_sd = process2(corr_sd)[0, 1:]
corr_nsd = process2(corr_nsd)[0, 1:]
slp_sd = np.asarray(slp_sd).mean(axis=0)
slp_nsd = np.asarray(slp_nsd).mean(axis=0)

time = np.linspace(0, 8, corr_sd.shape[0] + 1)[:-1]
width = np.diff(np.linspace(0, 8, corr_sd.shape[0] + 1))

# figure = Fig()
# fig, gs = figure.draw(num=1, grid=(4, 2))
axsd = plt.subplot(gs[3])
axslp_sd = axsd.twinx()
# axslp_sd.fill_between(
#     time, 0, slp_sd, color=sess.brainstates.colors["nrem"], alpha=0.5, zorder=1, ec=None
# )
axslp_sd.bar(
    time,
    gaussian_filter1d(slp_sd, sigma=1),
    zorder=1,
    color=sess.brainstates.colors["nrem"],
    alpha=0.4,
    width=width,
    edgecolor=None,
)

axsd.plot(time, corr_sd, "k", zorder=2)
axsd.spines["right"].set_visible(True)
axsd.set_ylabel("Correlation")
axsd.set_xlabel("Time (h)")

axnsd = plt.subplot(gs[4], sharey=axsd)
axslp_nsd = axnsd.twinx()
axslp_nsd.bar(
    time,
    gaussian_filter1d(slp_nsd, sigma=1),
    zorder=1,
    color=sess.brainstates.colors["nrem"],
    alpha=0.4,
    width=width,
    edgecolor=None,
)

axnsd.plot(time, corr_nsd, "k", zorder=2)
axnsd.set_xlabel("Time (h)")
axnsd.spines["right"].set_visible(True)
axslp_nsd.set_ylabel("nrem fraction")


figure.savefig("correlation", __file__)
# endregion

#%% Explained variance but by pooling all correlations into one vector
# region
sessions = subjects.Sd().ratSday3 + subjects.Nsd().ratSday2
# sessions = subjects.Sd().ratJday1
corr_all = pd.DataFrame()
for sub, sess in enumerate(sessions):

    pre = sess.epochs.pre
    maze = sess.epochs.maze
    if maze is None:
        maze = sess.epochs.maze1
    post = sess.epochs.post
    tag = sess.recinfo.animal.tag

    # --- break region into periods --------
    # bins = sess.utils.getinterval([post[0], post[0] + 5 * 3600], 2)
    # sess.spikes.stability.firingRate(periods=bins)
    spks = sess.spikes.times
    sd_period = [post[0], post[0] + 5 * 3600]
    intervals = sess.utils.getinterval(period=sd_period, nwindows=3)
    sess.spikes.stability.firingRate(periods=intervals)
    stability = sess.spikes.stability.info
    stable_pyr = np.where((stability.q < 4) & (stability.stable == 1))[0]
    spks = [spks[cell_id] for cell_id in stable_pyr]

    pre_corr = sess.spikes.corr.pairwise(spks, pre)
    maze_corr = sess.spikes.corr.pairwise(spks, maze)
    epochs = np.arange(post[0], post[0] + 8 * 3600 - 900, 300)
    corr = []
    for start in epochs:
        corr.append(sess.spikes.corr.pairwise(spikes=spks, period=[start, start + 900]))
    corr = np.asarray(corr)

    df = pd.DataFrame(corr.T, columns=[str(_) for _ in range(len(epochs))])
    df.insert(0, "maze", maze_corr)
    df.insert(0, "pre", pre_corr)
    df.insert(0, "grp", sess.recinfo.animal.tag)
    corr_all = corr_all.append(df)

figure = Fig()
fig, gs = figure.draw(num=1, grid=[4, 3])

gs = figure.subplot2grid(gs[2, :], grid=(1, 2))

for sub, grp in enumerate(["sd", "nsd"]):
    ev, rev = [], []
    data = corr_all[corr_all.grp == grp]
    for i in range(len(epochs)):
        par_corr = pg.partial_corr(data, x="maze", y=str(i), covar="pre")
        rev_par_corr = pg.partial_corr(data, x="maze", covar=str(i), y="pre")
        ev.append(par_corr.r2.values[0])
        rev.append(rev_par_corr.r2.values[0])

    ax = plt.subplot(gs[sub])
    t = np.linspace(0, 8, data.shape[1] - 3)
    ax.plot(t, ev, color=sess.replay.expvar.colors["ev"])
    ax.plot(t, rev, color=sess.replay.expvar.colors["rev"])
    ax.set_ylim([0, 0.25])
    ax.set_ylabel("Explained variance")
    ax.set_xlabel("Time (h)")

figure.savefig("exp_var", __file__)
# endregion


#%% Example figure for explained variance
# region

figure = Fig()
fig, gs = figure.draw(num=1, grid=(3, 2), wspace=0.3)

sessions = subjects.Sd().ratSday3

for sess in sessions:
    spikes = sess.spikes.pyr
    pre = sess.epochs.pre
    maze = sess.epochs.maze1
    post = sess.epochs.post
    precorr = sess.spikes.corr.pairwise(spikes, period=pre)
    mazecorr = sess.spikes.corr.pairwise(spikes, period=maze)
    postcorr = sess.spikes.corr.pairwise(spikes, period=[post[0], post[0] + 3600])

    ax = plt.subplot(gs[0])
    sns.regplot(
        x=mazecorr, y=precorr, ci=None, color="k", marker=".", line_kws={"color": "r"}
    )
    ax.set_xlabel("MAZE")
    ax.set_ylabel("PRE")

    ax = plt.subplot(gs[1], sharey=ax, sharex=ax)
    sns.regplot(
        x=mazecorr, y=postcorr, ci=None, color="k", marker=".", line_kws={"color": "r"}
    )
    # ax.scatter(mazecorr, postcorr, c="k", s=2)
    ax.set_xlabel("MAZE")
    ax.set_ylabel("POST")

# endregion


#%% CCG temporal structure over time
# region

figure = Fig()
fig, gs = figure.draw(num=1, grid=(2, 1))
sessions = subjects.Sd().ratSday3 + subjects.Nsd().ratSday2

for sub, sess in enumerate(sessions):
    spikes = sess.spikes.pyr
    maze = sess.epochs.maze1
    pre = sess.epochs.pre
    post = sess.epochs.post
    maze2 = sess.epochs.maze2
    post = [post[0], maze2[1]]
    maze = [post[0] + 4 * 3600, post[0] + 5 * 3600]

    def bin_spk(period):
        maze_spikes = [
            cell[np.where((cell > period[0]) & (cell < period[1]))[0]]
            for cell in spikes
        ]
        ccgs = sess.spikes.ccg_temporal(maze_spikes)
        return ccgs

    maze_corr = bin_spk(maze)
    pre_corr = bin_spk([pre[0], pre[0] + 900])

    df2 = pd.DataFrame({"maze": maze_corr, "pre": pre_corr})
    pre_ccg_corr = np.asarray(df2.corr())[0, 1]

    # indices = np.union1d(
    #     np.argwhere(np.isnan(maze_corr)), np.argwhere(np.isnan(pre_corr))
    # )
    # pre_ccg_corr = np.corrcoef(pre_corr[~indices], maze_corr[~indices])[0, 1]

    bins_period = np.arange(post[0], post[1], 900)

    corr_post = []
    for start in bins_period:
        post_corr = bin_spk([start, start + 900])
        df = pd.DataFrame({"maze": maze_corr, "post": post_corr})
        corr = np.asarray(df.corr())[0, 1]
        corr_post.append(corr)

    # ax = plt.subplot(gs[sub])
    gs_ = figure.subplot2grid(gs[sub], grid=(3, 1))
    ax = plt.subplot(gs_[1:])
    ax.plot((bins_period - post[0]) / 3600, corr_post, "k")
    # ax.axhline(pre_ccg_corr)
    ax.set_ylabel("correlation")
    ax.set_xlabel("Time (h)")
    # ax.set_ylim([-0.05, 0.29])

    axhypno = plt.subplot(gs_[0])
    sess.brainstates.hypnogram(ax=axhypno, tstart=post[0], unit="h")
    # a = np.corrcoef(maze_corr, pre_corr)[0, 1]
    # b = np.corrcoef(maze_corr, post_corr)[0, 1]

    # ax = plt.subplot(gs[0])
    # sns.regplot(x=maze_corr, y=pre_corr, ci=None)

    # ax = plt.subplot(gs[1])
    # sns.regplot(x=maze_corr, y=post_corr, ci=None)


# endregion

#%% Explained variance calculated with shuffled cell id
# region
figure = Fig()
fig, gs = figure.draw(num=1, grid=(3, 2))
sessions = (
    subjects.Nsd().ratSday2
    + subjects.Sd().ratSday3
    + subjects.Nsd().ratNday2
    + subjects.Sd().ratNday1
)
# sessions = subjects.Sd().ratSday3

for sub, sess in enumerate(sessions):

    pre = sess.epochs.pre

    try:
        maze = sess.epochs.maze
    except:
        maze = sess.epochs.maze1

    post = sess.epochs.post
    try:
        maze2 = sess.epochs.maze2
        post = [post[0], maze2[1]]
    except:
        post = post

    # --- break region into periods --------
    bins = sess.utils.getinterval([maze[0], post[1]], 5)
    sess.spikes.stability.firingRate(periods=bins)

    # template = maze
    # match = post

    sess.replay.expvar.compute_shuffle(
        template=maze, match=post, slideby=300, n_iter=100
    )
    ev = sess.replay.expvar.ev
    rev = sess.replay.expvar.rev

    axstate = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[sub], hspace=0.2)

    ax1 = fig.add_subplot(axstate[1:])
    sess.replay.expvar.plot(ax=ax1, tstart=post[0])
    ax1.set_xlim(left=0)
    ax1.tick_params(width=2)
    if sub == 3:
        ax1.set_ylim([0, 0.15])
    # # ax1.spines["right"].set_visible("False")
    # # ax1.spines["top"].set_visible("False")

    axhypno = fig.add_subplot(axstate[0], sharex=ax1)
    sess.brainstates.hypnogram(ax=axhypno, tstart=post[0], unit="h")
    # # panel_label(axhypno, "a")
    # # ax1.set_ylim([0, 11])

figure.savefig("expvar_cellid_shuffle")
# endregion


#%% Two-novel expvar
# region
figure = Fig()
fig, gs = figure.draw(num=1, grid=(2, 2))
sessions = subjects.Tn().ratSday5

for sub, sess in enumerate(sessions):

    pre = sess.epochs.pre

    maze1 = sess.epochs.maze1
    maze2 = sess.epochs.maze2
    post1 = sess.epochs.post1
    post2 = sess.epochs.post2

    # --- break region into periods --------
    bin1 = sess.utils.getinterval(pre, 2)
    bin2 = sess.utils.getinterval(post2, 3)
    bins = bin1 + bin2
    sess.spikes.stability.firingRate(periods=bins)

    # ----- expvar maze1 -------
    control = pre
    template = maze1  # [post[0] + 4 * 3600, post[0] + 5 * 3600]
    match = [post1[0], post2[1]]  # post2  # [post[0] + 5 * 3600, post[1]]

    sess.replay.expvar.compute(
        template=template,
        match=match,
        control=control,
        slideby=300,
        cross_shanks=True,
    )
    print(sess.replay.expvar.npairs)

    axstate = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[sub], hspace=0.2)

    ax1 = fig.add_subplot(axstate[1:])
    sess.replay.expvar.plot(ax=ax1, tstart=post1[0])
    ax1.set_xlim(left=0)
    ax1.tick_params(width=2)

    # ----- expvar maze2 -------
    control = pre
    template = maze2  # [post[0] + 4 * 3600, post[0] + 5 * 3600]
    match = post2  # [post[0] + 5 * 3600, post[1]]

    sess.replay.expvar.colors["ev"] = "#e75fdc"
    sess.replay.expvar.compute(
        template=template,
        match=match,
        control=control,
        slideby=300,
        cross_shanks=True,
    )
    print(sess.replay.expvar.npairs)

    sess.replay.expvar.plot(ax=ax1, tstart=post1[0])

    maze_time = (np.array(maze2) - post1[0]) / 3600
    ax1.axvspan(xmin=maze_time[0], xmax=maze_time[1], color="#f2a1a1", alpha=0.5)

ax1.legend(["rev_maze1", "maze1", "rev_maze2", "maze2"])
# figure.savefig("EV_sessions", __file__)

# endregion