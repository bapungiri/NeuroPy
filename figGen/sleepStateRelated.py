#%%
import numpy as np
from callfunc import processData
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.stats as stats
import pandas as pd
import seaborn as sns
import signal_process
import matplotlib as mpl

basePath = [
    "/data/Clustering/SleepDeprivation/RatJ/Day1/",
    "/data/Clustering/SleepDeprivation/RatK/Day1/",
    "/data/Clustering/SleepDeprivation/RatN/Day1/",
    "/data/Clustering/SleepDeprivation/RatJ/Day2/",
    "/data/Clustering/SleepDeprivation/RatK/Day2/",
    "/data/Clustering/SleepDeprivation/RatN/Day2/",
]


sessions = [processData(_) for _ in basePath]

plt.clf()
fig = plt.figure(1, figsize=(10, 15))
gs = gridspec.GridSpec(4, 4, figure=fig)
fig.subplots_adjust(hspace=0.3, wspace=0.3)
fig.suptitle("Sleep states related analysis")
titlesize = 8


#%% Spectrogram example
# region
for sub, sess in enumerate([sessions[2]]):

    sess.trange = np.array([])
    t_start = sess.epochs.post[0] + 5 * 3600
    axstate = gridspec.GridSpecFromSubplotSpec(5, 1, subplot_spec=gs[0, :], hspace=0.1)
    axspec = fig.add_subplot(axstate[1:4])
    sess.viewdata.specgram(ax=axspec)

    axhypno = fig.add_subplot(axstate[0], sharex=axspec)
    sess.viewdata.hypnogram(ax1=axhypno)

    axhypno = fig.add_subplot(axstate[-1], sharex=axspec)
    t = sess.brainstates.params.time
    emg = sess.brainstates.params.emg
    axhypno.plot(t, emg, "#4a4e68")

# endregion


#%% NREM,REM Duration compare between SD and control
# region
group = []
for sub, sess in enumerate(sessions):

    sess.trange = np.array([])
    if sub < 3:
        t_start = sess.epochs.post[0] + 5 * 3600
        condition = "SD"
    else:
        t_start = sess.epochs.post[0]
        condition = "NSD"

    df = sess.brainstates.states
    df = df.loc[((df["state"] == 2) | (df["state"] == 1)) & (df["start"] > t_start)]
    df["condition"] = [condition] * len(df)
    group.append(df)

group = pd.concat(group, ignore_index=True)
ax = fig.add_subplot(gs[1, 0])
sns.boxplot(x="state", y="duration", hue="condition", data=group, palette="Set3", ax=ax)
ax.set_ylim(-10, 2000)
ax.set_ylabel("duration (s)")
ax.set_xlabel("")
ax.set_xticklabels(["nrem", "rem"])
# endregion

#%% Mean theta-delta ratio compare between SD and control
# region
# plt.clf()
group = []
for sub, sess in enumerate(sessions):

    sess.trange = np.array([])
    if sub < 3:
        t_start = sess.epochs.post[0] + 5 * 3600
        condition = "SD"
    else:
        t_start = sess.epochs.post[0]
        condition = "NSD"

    states = sess.brainstates.states
    states = states.loc[(states["name"] == "rem") & (states["start"] > t_start)]
    states["condition"] = [condition] * len(states)

    params = sess.brainstates.params
    theta_delta = []
    for epoch in states.itertuples():
        val = params.loc[
            (params["time"] > epoch.start) & (params["time"] < epoch.end),
            "theta_delta_ratio",
        ]
        theta_delta.append(np.mean(val))

    states["theta_delta"] = theta_delta
    group.append(states)


group = pd.concat(group, ignore_index=True)
ax = fig.add_subplot(gs[1, 1])
sns.boxplot(x="condition", y="theta_delta", data=group, palette="Set3", ax=ax)
ax.set_ylabel("ratio")
ax.set_xlabel("")
ax.set_title("Mean theta-delta ratio \n during REM", fontsize=titlesize)
# ax.set_xticklabels(["sd", "rem"])
# ax.legend("")
# endregion


#%% Delta amplitude 1st NREM
# region
group = []
for sub, sess in enumerate(sessions):

    sess.trange = np.array([])
    if sub < 3:
        t_start = sess.epochs.post[0] + 5 * 3600
        condition = "SD"
    else:
        t_start = sess.epochs.post[0]
        condition = "NSD"

    states = sess.brainstates.states
    states = states.loc[
        (states["name"] == "nrem")
        & (states["start"] > t_start)
        & (states["duration"] > 300)
    ]
    states["condition"] = [condition] * len(states)

    params = sess.brainstates.params
    first_nrem = states[:1].reset_index()
    val = params.loc[
        (params["time"] > first_nrem.start[0]) & (params["time"] < first_nrem.end[0]),
        "delta",
    ]
    first_nrem["delta"] = np.mean(val)
    group.append(first_nrem)


group = pd.concat(group, ignore_index=True)
ax = fig.add_subplot(gs[1, 2])
sns.boxplot(x="condition", y="delta", data=group, palette="Set3", ax=ax)
ax.set_ylabel("amplitude")
ax.set_xlabel("")
ax.set_title("Delta band Power 1st NREM \n (>5 minutes)", fontsize=titlesize)
# ax.set_xticklabels(["nrem"])
# endregion

#%% Ripple band amplitude 1st NREM
# region
group = []
for sub, sess in enumerate(sessions):

    sess.trange = np.array([])
    if sub < 3:
        t_start = sess.epochs.post[0] + 5 * 3600
        condition = "SD"
    else:
        t_start = sess.epochs.post[0]
        condition = "NSD"

    states = sess.brainstates.states
    states = states.loc[
        (states["name"] == "nrem")
        & (states["start"] > t_start)
        & (states["duration"] > 300)
    ]
    states["condition"] = [condition] * len(states)

    params = sess.brainstates.params
    first_nrem = states[:1].reset_index()
    val = params.loc[
        (params["time"] > first_nrem.start[0]) & (params["time"] < first_nrem.end[0]),
        "ripple",
    ]
    first_nrem["ripple"] = np.mean(val)
    group.append(first_nrem)


group = pd.concat(group, ignore_index=True)
ax = fig.add_subplot(gs[1, 3])
sns.boxplot(x="condition", y="ripple", data=group, palette="Set3", ax=ax)
ax.set_ylabel("amplitude")
ax.set_xlabel("")
ax.set_title("Ripple band Power 1st NREM \n (>5 minutes)", fontsize=titlesize)
# ax.set_xticklabels(["nrem"])
# endregion


#%% Powers at various bands scross sleep deprivation
# region
group = []
for sub, sess in enumerate(sessions[:3]):

    sess.trange = np.array([])
    lfp = sess.spindle.best_chan_lfp()[0]
    eegSrate = sess.recinfo.lfpSrate
    post = sess.epochs.post
    sd_period = [post[0], post[0] + 5 * 3600]
    t = np.linspace(0, len(lfp) / eegSrate, len(lfp))

    lfpsd = lfp[(t > sd_period[0]) & (t < sd_period[1])]
    tsd = np.linspace(sd_period[0], sd_period[1], len(lfpsd))
    binsd = np.linspace(sd_period[0], sd_period[1], 6)

    specgram = signal_process.spectrogramBands(lfpsd)
    bands = [
        specgram.delta,
        specgram.theta,
        specgram.spindle,
        specgram.gamma,
        specgram.ripple,
    ]

    mean_bands = stats.binned_statistic(
        specgram.time + sd_period[0], bands, statistic="mean", bins=binsd
    )

    mean_bands = mean_bands.statistic.T / np.sum(mean_bands.statistic, axis=1)

    df = pd.DataFrame(
        mean_bands, columns=["delta", "theta", "spindle", "gamma", "ripple"]
    )
    subname = sess.sessinfo.session.sessionName
    df["subject"] = [subname] * len(df)
    df["hour"] = np.arange(1, 6)
    group.append(df)


group = pd.concat(group, ignore_index=True)
group_long = pd.melt(
    group, id_vars=["hour", "subject"], var_name=["bands"], value_name="amplitude"
)

cmap = mpl.cm.get_cmap("Set3")
colors = [cmap(ind) for ind in range(5)]
colors = np.asarray(list(np.concatenate([[col] * 5 for col in colors])))
ax = fig.add_subplot(gs[2, 2:4])
ax.clear()
sns.barplot(
    x="bands",
    y="amplitude",
    hue="hour",
    data=group_long,
    # palette="Set3",
    color=colors[0],
    # edgecolor=".05",
    errwidth=1,
    # ax=ax,
)
ax.set_ylabel("Normalized amplitude")
# ax.legend(ncol=5)
ax.legend("")
# ax.set_xlabel("")
ax.set_title("Band power during SD (hourly, 5 hours)", fontsize=titlesize)
fig.show()
# endregion
