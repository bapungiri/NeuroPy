#%%
import numpy as np
from callfunc import processData
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.stats as stats
import scipy.signal as sg
import pandas as pd
import seaborn as sns
import signal_process
import matplotlib as mpl
import warnings

warnings.simplefilter(action="default")


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
fig = plt.figure(1, figsize=(8.5, 11))
gs = gridspec.GridSpec(4, 4, figure=fig)
fig.subplots_adjust(hspace=0.3, wspace=0.5)
fig.suptitle("Sleep states related analysis")
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


#%% Spectrogram example
# region
for sub, sess in enumerate([sessions[2]]):

    sess.trange = np.array([])
    t_start = sess.epochs.post[0] + 5 * 3600
    t = sess.brainstates.params.time
    emg = sess.brainstates.params.emg
    delta = sess.brainstates.params.delta

    axstate = gridspec.GridSpecFromSubplotSpec(6, 1, subplot_spec=gs[0, :], hspace=0.2)
    axspec = fig.add_subplot(axstate[1:4])
    sess.viewdata.specgram(ax=axspec)
    axspec.axes.get_xaxis().set_visible(False)

    axdelta = fig.add_subplot(axstate[4], sharex=axspec)
    axdelta.fill_between(t, 0, delta, color="#9E9E9E")
    axdelta.axes.get_xaxis().set_visible(False)
    axdelta.set_ylabel("Delta")

    axhypno = fig.add_subplot(axstate[0], sharex=axspec)
    sess.viewdata.hypnogram(ax1=axhypno)
    panel_label(axhypno, "a")

    axemg = fig.add_subplot(axstate[-1], sharex=axspec)
    axemg.plot(t, emg, "#4a4e68", lw=0.8)
    axemg.set_ylabel("EMG")
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
    df = df.loc[
        ((df.name == "rem") | (df.name == "nrem")) & (df.start > t_start)
    ].copy()
    df["condition"] = [condition] * len(df)
    group.append(df)

group = pd.concat(group, ignore_index=True)
ax = fig.add_subplot(gs[1, 0])
ax.clear()
sns.boxplot(x="state", y="duration", hue="condition", data=group, palette="Set3", ax=ax)
ax.set_ylim(-10, 2000)
ax.set_ylabel("duration (s)")
ax.set_xlabel("")
ax.set_xticklabels(["nrem", "rem"])
panel_label(ax, "b")
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
    states = states.loc[(states.name == "rem") & (states.start > t_start)].copy()

    params = sess.brainstates.params
    theta_delta = []
    for epoch in states.itertuples():
        val = params.loc[
            (params.time > epoch.start) & (params.time < epoch.end),
            "theta_deltaplus_ratio",
        ]
        theta_delta.append(np.mean(val))

    states.loc[:, "theta_delta"] = theta_delta
    states.loc[:, "condition"] = [condition] * len(states)
    group.append(states)


group = pd.concat(group, ignore_index=True)
ax = fig.add_subplot(gs[1, 1])
ax.clear()
sns.boxplot(
    x="condition", y="theta_delta", data=group, palette="Set3", ax=ax, width=0.5
)
ax.set_ylabel("ratio")
ax.set_xlabel("")
ax.set_title("Mean theta-delta ratio \n during REM", fontsize=titlesize)
panel_label(ax, "c")

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
        (states.name == "nrem") & (states.start > t_start) & (states.duration > 300)
    ].copy()
    states["condition"] = [condition] * len(states)

    params = sess.brainstates.params
    first_nrem = states[:1].reset_index()
    val = params.loc[
        (params["time"] > first_nrem.start[0]) & (params["time"] < first_nrem.end[0]),
        "delta",
    ].copy()
    first_nrem["delta"] = np.mean(val)
    group.append(first_nrem)


group = pd.concat(group, ignore_index=True)
ax = fig.add_subplot(gs[1, 2])
ax.clear()
sns.boxplot(x="condition", y="delta", data=group, palette="Set3", ax=ax, width=0.5)
ax.set_ylabel("amplitude")
ax.set_xlabel("")
ax.set_title("Delta Power 1st NREM \n (>5 minutes)", fontsize=titlesize)
# ax.set_xticklabels(["nrem"])
panel_label(ax, "d")
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
    ].copy()
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
ax.clear()
sns.boxplot(x="condition", y="ripple", data=group, palette="Set3", ax=ax, width=0.5)
ax.set_ylabel("amplitude")
ax.set_xlabel("")
ax.set_title("Ripple Power 1st NREM \n (>5 minutes)", fontsize=titlesize)
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
panel_label(ax, "e")

# ax.set_xticklabels(["nrem"])
# endregion

#%% PSD first hour vs last hour and plotting the difference
# region

psd1st_all, psd5th_all, psd_diff = [], [], []
for sub, sess in enumerate(sessions[:3]):
    sess.trange = np.array([])
    eegSrate = sess.recinfo.lfpSrate
    channel = sess.theta.bestchan
    post = sess.epochs.post
    eeg = sess.utils.geteeg(chans=channel, timeRange=[post[0], post[0] + 5 * 3600])
    nfrms_hour = eegSrate * 3600
    lfp1st = eeg[:nfrms_hour]
    lfp5th = eeg[-nfrms_hour:]

    psd = lambda sig: sg.welch(
        sig, fs=eegSrate, nperseg=10 * eegSrate, noverlap=5 * eegSrate
    )
    multitaper = lambda sig: signal_process.mtspect(
        sig, fs=eegSrate, nperseg=10 * eegSrate, noverlap=5 * eegSrate
    )

    _, psd1st = multitaper(lfp1st)
    f, psd5th = multitaper(lfp5th)

    psd1st_all.append(psd1st)
    psd5th_all.append(psd5th)
    psd_diff.append(psd1st - psd5th)

psd1st_all = np.asarray(psd1st_all).mean(axis=0)
psd5th_all = np.asarray(psd5th_all).mean(axis=0)
psd_diff = np.asarray(psd_diff).mean(axis=0)

ax = fig.add_subplot(gs[2, 1])
ax.clear()
# ax.plot(f, psd1st_all, "k", label="ZT1")
# ax.plot(f, psd5th_all, "r", label="ZT5")
ax.plot(stats.zscore(psd_diff), "k")
ax.set_xscale("log")
ax.set_xlim([1, 300])
# ax.set_ylim([10, 10e5])
ax.set_title("PSDZT1-PSDZT5", fontsize=titlesize)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Zscore of difference")
# ax.legend()
panel_label(ax, "f")

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
    ci="sd",
)
ax.set_ylabel("Normalized amplitude")
# ax.legend(ncol=5)
ax.legend("")
ax.set_xlabel("")
ax.set_title("Band power during SD (hourly, 5 hours)", fontsize=titlesize)
fig.show()
panel_label(ax, "g")

# endregion

