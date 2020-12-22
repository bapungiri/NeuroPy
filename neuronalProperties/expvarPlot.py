#%% Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter
import subjects
from callfunc import processData
from plotUtil import Fig

#%% Explained variance PRE-MAZE-POST
# region
"""Only found stable units for 3 sessions
"""
figure = Fig()
fig, gs = figure.draw(grid=(1, 1))
sessions = subjects.sd([3])

for sub, sess in enumerate(sessions):

    sess.trange = np.array([])
    pre = sess.epochs.pre
    maze = sess.epochs.maze1
    post = sess.epochs.post
    bins = [
        pre,
        [post[0] + 4 * 3600, post[0] + 5 * 3600],
        [post[0] + 5 * 3600, post[0] + 10 * 4600],
        # post,
        # [post[0] + 5 * 3600, post[0] + 8 * 3600],
    ]
    sess.spikes.stability.firingRate(bins=bins)
    # sess.spikes.stability.refPeriodViolation()
    # violations = sess.spikes.stability.violations
    sess.replay.expvar.compute(template=bins[1], match=bins[2], control=bins[0])

    axstate = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[sub], hspace=0.2)

    ax1 = fig.add_subplot(axstate[1:])
    sess.replay.expvar.plot(ax=ax1, tstart=bins[2][0])
    # ax1.spines["right"].set_visible("False")
    # ax1.spines["top"].set_visible("False")

    axhypno = fig.add_subplot(axstate[0], sharex=ax1)
    sess.brainstates.hypnogram(ax1=axhypno, tstart=bins[2][0], unit="h")
    # panel_label(axhypno, "a")
    # ax1.set_ylim([0, 11])


# endregion

#%% Explained variance during recovery sleep while controlling for MAZE correlations
# region
"""Only found stable units for 3 sessions
"""
plt.clf()
fig = plt.figure(1, figsize=(6, 10))
gs = gridspec.GridSpec(3, 5, figure=fig)
fig.subplots_adjust(hspace=0.4, wspace=0.0)

for sub, sess in enumerate(sessions):

    sess.trange = np.array([])
    maze = sess.epochs.maze
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
        axhypno.set_title(sess.sessinfo.session.sessionName)
    # panel_label(axhypno, "a")
    # ax1.set_ylim([0, 0.3])


# endregion
