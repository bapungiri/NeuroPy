#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter
from ccg import correlograms
from mathutil import getICA_Assembly
from callfunc import processData
import subjects
from plotUtil import Fig

#%% Reactivation strength
plt.clf()
fig = plt.figure(1, figsize=(10, 15))
gs = gridspec.GridSpec(16, 1, figure=fig)
fig.subplots_adjust(hspace=0.3)
sessions = subjects.Sd().ratSday3
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


#%% similarity of ICA vectors detected during MAZE and last hour of sleep deprivation
# region
plt.clf()
fig = plt.figure(1, figsize=(10, 15))
gs = gridspec.GridSpec(3, 3, figure=fig)
fig.subplots_adjust(hspace=0.3)

for sub, sess in enumerate(sessions):
    sess.trange = np.array([])
    spikes = sess.spikes.times
    pre = sess.epochs.pre
    maze = sess.epochs.maze
    post = sess.epochs.post
    bins = [
        [maze[0], maze[1]],
        [post[0], post[0] + 3600],
        [post[0] + 4 * 3600, post[0] + 5 * 3600],
        # [post[0] + 5 * 3600, post[0] + 6 * 3600],
    ]
    sess.spikes.stability.firingRate(bins=bins)
    stability = sess.spikes.stability.info
    stable_pyr = np.where((stability.q < 4) & (stability.stable == 1))[0]

    pyr = [spikes[cell_id] for cell_id in stable_pyr]
    maze_bin = np.arange(bins[0][0], bins[0][1], 0.25)
    spkcnt_maze = np.asarray([np.histogram(cell, bins=maze_bin)[0] for cell in pyr])
    sd_bin = np.arange(bins[1][0], bins[1][1], 0.25)
    spkcnt_zt5 = np.asarray([np.histogram(cell, bins=sd_bin)[0] for cell in pyr])
    rec_bin = np.arange(bins[2][0], bins[2][1], 0.25)
    spkcnt_rec = np.asarray([np.histogram(cell, bins=rec_bin)[0] for cell in pyr])

    v_maze = sess.replay.assemblyICA.getAssemblies(spkcnt_maze)
    v_sd = sess.replay.assemblyICA.getAssemblies(spkcnt_zt5)
    v_rec = sess.replay.assemblyICA.getAssemblies(spkcnt_rec)

    # v_maze = np.where(v_maze > 0, 1, 0)
    # v_sd = np.where(v_sd > 0, 1, 0)
    # v_rec = np.where(v_rec > 0, 1, 0)

    ica_similarity_maze_rec = np.abs(np.matmul(v_maze.T, v_sd))
    ica_similarity_sd_rec = np.abs(np.matmul(v_maze.T, v_rec))

    ax = fig.add_subplot(gs[sub, 0])
    ax.bar([1, 2], [np.mean(ica_similarity_maze_rec), np.mean(ica_similarity_sd_rec)])

    ax = fig.add_subplot(gs[sub, 1])
    ax.imshow(np.abs(ica_similarity_maze_rec), vmax=1, vmin=0)

    ax = fig.add_subplot(gs[sub, 2])
    ax.imshow(np.abs(ica_similarity_sd_rec), vmax=1, vmin=0)


# endregion

#%% Activations strength during sleep deprivation
# region

plt.clf()
fig = plt.figure(1, figsize=(10, 15))
gs = gridspec.GridSpec(1, 3, figure=fig)
fig.subplots_adjust(hspace=0.3, wspace=0.2)
for sub, sess in enumerate(sessions[:3]):
    sess.trange = np.array([])
    maze = sess.epochs.maze
    post = sess.epochs.post
    spks = sess.spikes.times
    sd_period = [maze[0], post[0] + 5 * 3600]
    intervals = sess.utils.getinterval(period=sd_period, nwindows=5)
    sess.spikes.stability.firingRate(bins=intervals)
    stability = sess.spikes.stability.info
    stable_pyr = np.where((stability.q < 4) & (stability.stable == 1))[0]

    pyr = [spks[cell_id] for cell_id in stable_pyr]
    print(len(pyr))
    # sd_bin = np.arange(post[0], post[0] + 5 * 3600, 0.25)
    # spkcnt_sd = np.array([np.histogram(cell, bins=sd_bin)[0] for cell in pyr])

    activation, match_bin = sess.replay.assemblyICA.getActivation(
        template=maze, match=[sd_period[0], sd_period[1]], spks=pyr
    )

    ax = plt.subplot(gs[sub])
    sess.replay.assemblyICA.plotActivation(ax=ax)
    ax.set_title(sess.sessinfo.session.sessionName)

# endregion

#%% swr vs reactivation strength during sleep deprivation
# region

plt.clf()
fig = plt.figure(1, figsize=(10, 15))
gs = gridspec.GridSpec(1, 4, figure=fig)
fig.subplots_adjust(hspace=0.3, wspace=0.2)
for sub, sess in enumerate(sessions):
    sess.trange = np.array([])
    maze = sess.epochs.maze
    post = sess.epochs.post
    spks = sess.spikes.times
    stability_period = [maze[0], post[0] + 5 * 3600]
    intervals = sess.utils.getinterval(period=stability_period, nwindows=5)
    sess.spikes.stability.firingRate(bins=intervals)
    stability = sess.spikes.stability.info
    stable_pyr = np.where((stability.q < 4) & (stability.stable == 1))[0]

    pyr = [spks[cell_id] for cell_id in stable_pyr]
    print(len(pyr))

    sd_period = [post[0], post[0] + 5 * 3600]
    activation, match_bin = sess.replay.assemblyICA.getActivation(
        template=maze, match=[sd_period[0], sd_period[1]], spks=pyr
    )

    # sess.replay.assemblyICA.plotActivation(ax=plt.subplot(gs[sub]))

    sum_activation = np.sum(stats.zscore(activation, axis=1), axis=0)

    ripple = sess.ripple.time
    ripple_zt1 = ripple[
        (ripple[:, 0] > sd_period[0]) & (ripple[:, 0] < sd_period[0] + 3600), :
    ]
    ripple_zt5 = ripple[
        (ripple[:, 0] > sd_period[0] + 4 * 3600)
        & (ripple[:, 0] < sd_period[0] + 5 * 3600),
        :,
    ]

    mean_react_zt1 = stats.binned_statistic(
        match_bin[1:] - 0.125, sum_activation, bins=np.concatenate(ripple_zt1)
    )[0][::2]
    mean_react_zt5 = stats.binned_statistic(
        match_bin[1:] - 0.125, sum_activation, bins=np.concatenate(ripple_zt5)
    )[0][::2]

    ax = plt.subplot(gs[sub])
    ax.bar([1, 2], [np.nanmean(mean_react_zt1), np.nanmean(mean_react_zt5)])
# endregion

#%% delta vs reactivation strength duriing sleep deprivation
# region

plt.clf()
fig = plt.figure(1, figsize=(10, 15))
gs = gridspec.GridSpec(1, 4, figure=fig)
fig.subplots_adjust(hspace=0.3, wspace=0.2)
for sub, sess in enumerate(sessions):
    sess.trange = np.array([])
    maze = sess.epochs.maze
    post = sess.epochs.post
    spks = sess.spikes.times
    stability_period = [maze[0], post[0] + 5 * 3600]
    intervals = sess.utils.getinterval(period=stability_period, nwindows=5)
    sess.spikes.stability.firingRate(bins=intervals)
    stability = sess.spikes.stability.info
    stable_pyr = np.where((stability.q < 4) & (stability.stable == 1))[0]

    pyr = [spks[cell_id] for cell_id in stable_pyr]
    print(len(pyr))

    sd_period = [post[0], post[0] + 5 * 3600]
    activation, match_bin = sess.replay.assemblyICA.getActivation(
        template=maze, match=[sd_period[0], sd_period[1]], spks=pyr
    )

    # sess.replay.assemblyICA.plotActivation(ax=plt.subplot(gs[sub]))

    sum_activation = np.sum(stats.zscore(activation, axis=1), axis=0)

    ripple = sess.swa.time
    ripple_zt1 = ripple[
        (ripple[:, 0] > sd_period[0]) & (ripple[:, 0] < sd_period[0] + 3600), :
    ]
    ripple_zt5 = ripple[
        (ripple[:, 0] > sd_period[0] + 4 * 3600)
        & (ripple[:, 0] < sd_period[0] + 5 * 3600),
        :,
    ]

    mean_react_zt1 = stats.binned_statistic(
        match_bin[1:] - 0.125, sum_activation, bins=np.concatenate(ripple_zt1)
    )[0][::2]
    mean_react_zt5 = stats.binned_statistic(
        match_bin[1:] - 0.125, sum_activation, bins=np.concatenate(ripple_zt5)
    )[0][::2]

    ax = plt.subplot(gs[sub])
    ax.bar([1, 2], [np.nanmean(mean_react_zt1), np.nanmean(mean_react_zt5)])
# endregion


#%% Firing rate before and after swr comparison between sleep and swr
# region

plt.clf()
fig = plt.figure(1, figsize=(10, 15))
gs = gridspec.GridSpec(1, 4, figure=fig)
fig.subplots_adjust(hspace=0.3, wspace=0.2)
for sub, sess in enumerate(sessions):
    sess.trange = np.array([])
    maze = sess.epochs.maze
    post = sess.epochs.post
    spks = sess.spikes.times
    stability_period = [post[0], post[0] + 5 * 3600]
    intervals = sess.utils.getinterval(period=stability_period, nwindows=5)
    sess.spikes.stability.firingRate(bins=intervals)
    stability = sess.spikes.stability.info
    stable_pyr = np.where((stability.q < 4) & (stability.stable == 1))[0]

    pyr = [spks[cell_id] for cell_id in stable_pyr]
    print(len(pyr))
    spks = np.concatenate(pyr)

    sd_period = [post[0], post[0] + 5 * 3600]

    ripple = sess.ripple.time
    ripple_sd = ripple[(ripple[:, 0] > sd_period[0]) & (ripple[:, 0] < sd_period[1]), :]

    instfiring = sess.spikes.instfiring

    instfire_rpl = np.zeros(1999)
    for rpl in ripple_sd:
        instfire_rpl += instfiring.loc[
            (instfiring.time > rpl[0] - 1) & (instfiring.time < rpl[0] + 1), "frate"
        ].to_numpy()[:1999]

    ax = plt.subplot(gs[sub])
    ax.plot(instfire_rpl / len(ripple_sd))

# endregion

#%% Directed reward vs sprinkled reward reactivation comparison
# region
# figure = Fig()
# fig, gs = figure.draw(num=2, grid=(1, 1))
sessions = subjects.Of().ratNday4
for sub, sess in enumerate(sessions):
    maze = sess.epochs.maze
    sprinkle = sess.epochs.sprinkle
    pre_sprinkle = [maze[0], sprinkle[0]]
    activation = sess.replay.assemblyICA.getActivation(
        template=pre_sprinkle, match=maze
    )
    # ax = plt.subplot(gs[0])
    sess.replay.assemblyICA.plotActivation()

# endregion