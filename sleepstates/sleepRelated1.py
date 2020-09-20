#%%
import os
import warnings

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as sg
import scipy.stats as stats
from scipy.stats.stats import gstd
import seaborn as sns

import signal_process
from callfunc import processData
from plotUtil import Fig

warnings.simplefilter(action="default")


#%% functions
# region
def scale(x):

    x = x - np.min(x)
    x = x / np.max(x)

    return x


def getPxx(lfp):
    window = 5 * 1250

    freq, Pxx = sg.welch(
        lfp, fs=1250, nperseg=window, noverlap=window / 6, detrend="linear",
    )
    noise = np.where(
        ((freq > 59) & (freq < 61)) | ((freq > 119) & (freq < 121)) | (freq > 220)
    )[0]
    freq = np.delete(freq, noise)
    Pxx = np.delete(Pxx, noise)

    return Pxx, freq


# endregion


basePath = [
    "/data/Clustering/SleepDeprivation/RatJ/Day1/",
    "/data/Clustering/SleepDeprivation/RatK/Day1/",
    "/data/Clustering/SleepDeprivation/RatN/Day1/",
    "/data/Clustering/SleepDeprivation/RatJ/Day2/",
    "/data/Clustering/SleepDeprivation/RatK/Day2/",
    "/data/Clustering/SleepDeprivation/RatN/Day2/",
]


sessions = [processData(_) for _ in basePath]


#%% Spectrogram example
# region
figure = Fig()
fig, gs = figure.draw(grid=[4, 4])

axstate = gridspec.GridSpecFromSubplotSpec(6, 1, subplot_spec=gs[0, :], hspace=0.2)

for sub, sess in enumerate(sessions[2:3]):

    sess.trange = np.array([])
    t_start = sess.epochs.post[0] + 5 * 3600
    t = sess.brainstates.params.time
    emg = sess.brainstates.params.emg
    delta = sess.brainstates.params.delta

    axspec = fig.add_subplot(axstate[1:4])
    sess.viewdata.specgram(ax=axspec)
    axspec.axes.get_xaxis().set_visible(False)

    axdelta = fig.add_subplot(axstate[4], sharex=axspec)
    axdelta.fill_between(t, 0, delta, color="#9E9E9E")
    axdelta.axes.get_xaxis().set_visible(False)
    axdelta.set_ylabel("Delta")

    axhypno = fig.add_subplot(axstate[0], sharex=axspec)
    sess.viewdata.hypnogram(ax1=axhypno)
    figure.panel_label(axhypno, "a")

    axemg = fig.add_subplot(axstate[-1], sharex=axspec)
    axemg.plot(t, emg, "#4a4e68", lw=0.8)
    axemg.set_ylabel("EMG")

figure.savefig("spectrogram_example_sd", __file__)
# endregion


#%% NREM,REM Duration compare between SD and control
# region
figure = Fig()
fig, gs = figure.draw(grid=[4, 4])
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
figure.panel_label(ax, "b")
figure.savefig("nrem_rem_duration_compare", __file__)
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
"""We want to understand the changes in spectral power across sleep deprivation, one interesting way to look at that is plotting the difference of power across frequencies.
"""
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

# TODO add distributions of emg changes across the SD period
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


scriptname = os.path.basename(__file__)
filename = "Test"
savefig(fig, filename, scriptname)
# endregion


#%% Powerspectrum compare during REM
# region
for sub, sess in enumerate(sessions):

    sess.trange = np.array([])

    sampfreq = sess.recinfo.lfpSrate
    maze = sess.epochs.maze
    tstart = sess.epochs.post[0]

    lfp, _, _ = sess.spindle.best_chan_lfp()
    t = np.linspace(0, len(lfp) / 1250, len(lfp))
    deadfile = sess.sessinfo.files.filePrefix.with_suffix(".dead")
    if deadfile.is_file():
        with deadfile.open("r") as f:
            noisy = []
            for line in f:
                epc = line.split(" ")
                epc = [float(_) for _ in epc]
                noisy.append(epc)
            noisy = np.asarray(noisy)
            noisy = ((noisy / 1000) * sampfreq).astype(int)

        for noisy_ind in range(noisy.shape[0]):
            st = noisy[noisy_ind, 0]
            en = noisy[noisy_ind, 1]
            numnoisy = en - st
            lfp[st:en] = np.nan

    states = sess.brainstates.states
    rem = states[(states["start"] > tstart) & (states["name"] == "rem")]

    binlfp = lambda x, t1, t2: x[(t > t1) & (t < t2)]
    lfprem = []
    for epoch in rem.itertuples():
        lfprem.extend(binlfp(lfp, epoch.start, epoch.end))
    lfprem = stats.zscore(np.asarray(lfprem))

    lfpmaze = stats.zscore(binlfp(lfp, maze[0], maze[1]))
    # b, a = sg.iirnotch(60, 30, fs=1250)
    # lfprem = sg.filtfilt(b, a, lfprem)

    sess.pxx_rem, sess.f_rem = getPxx(lfprem)
    sess.pxx_maze, sess.f_maze = getPxx(lfpmaze)


# ====== Plotting ==========
plt.clf()
fig = plt.figure(1, figsize=(15, 8))
gs = GridSpec(1, 3, figure=fig)
# fig.subplots_adjust(hspace=0.5)

for sub, sess in enumerate(sessions):

    if sub < 3:
        plt_ind = sub
        alpha = 1
        shift = 0
    else:
        plt_ind = sub - 3
        alpha = 0.4
        color = "k"
        shift = 2

    subname = sess.sessinfo.session.name

    todB = lambda power: 10 * np.log10(power)

    ax = fig.add_subplot(gs[0, plt_ind])
    plt.plot(sess.f_rem, todB(sess.pxx_rem) - shift, color="#ef253c", alpha=alpha)
    plt.plot(sess.f_maze, todB(sess.pxx_maze) - shift + 6, color=color, alpha=alpha)
    plt.xscale("log")
    # plt.yscale("log")
    ax.set_xlim([4, 220])
    # ax.set_xlim([4, 220])
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("Power (dB)")
    ax.set_title(subname)


ax.legend(["REM-SD", "MAZE-SD", "REM-NSD", "MAZE-NSD"])
fig.suptitle(
    f"Power spectrum REM and MAZE epochs between SD and NSD session. Only REM periods in POST are compared. \n Note: curves have been artificially shifted for clarity"
)
# endregion

#%% Bicoherence plots of REM sleep
# region
plt.clf()
fig = plt.figure(1, figsize=(10, 15))
gs = gridspec.GridSpec(2, 3, figure=fig)
fig.subplots_adjust(hspace=0.3)
for sub, sess in enumerate(sessions):

    sess.trange = np.array([])

    sampfreq = sess.recinfo.lfpSrate
    maze = sess.epochs.maze
    tstart = sess.epochs.post[0]

    lfp, _, _ = sess.spindle.best_chan_lfp()
    t = np.linspace(0, len(lfp) / 1250, len(lfp))
    deadfile = sess.sessinfo.files.filePrefix.with_suffix(".dead")
    if deadfile.is_file():
        with deadfile.open("r") as f:
            noisy = []
            for line in f:
                epc = line.split(" ")
                epc = [float(_) for _ in epc]
                noisy.append(epc)
            noisy = np.asarray(noisy)
            noisy = ((noisy / 1000) * sampfreq).astype(int)

        for noisy_ind in range(noisy.shape[0]):
            st = noisy[noisy_ind, 0]
            en = noisy[noisy_ind, 1]
            numnoisy = en - st
            lfp[st:en] = np.nan

    states = sess.brainstates.states
    rem = states[(states["start"] > tstart) & (states["name"] == "rem")]

    binlfp = lambda x, t1, t2: x[(t > t1) & (t < t2)]
    lfprem = []
    for epoch in rem.itertuples():
        lfprem.extend(binlfp(lfp, epoch.start, epoch.end))
    lfprem = stats.zscore(np.asarray(lfprem))

    # strong_theta = strong_theta - np.mean(strong_theta)
    lfprem = sg.detrend(lfprem, type="linear")
    bicoh, bicoh_freq = signal_process.bicoherence(
        lfprem, window=4 * 1250, overlap=2 * 1250
    )

    ax = fig.add_subplot(gs[sub])
    ax.pcolorfast(bicoh_freq, bicoh_freq, bicoh, cmap="YlGn", vmax=0.2)
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Frequency (Hz)")
    # plt.pcolormesh(bispec_freq, bispec_freq, bispec, vmin=0, vmax=0.1, cmap="YlGn")
    ax.set_ylim([2, 75])

    # ax = fig.add_subplot(gs[sub + 2])
    # f, t, sxx = sg.spectrogram(strong_theta, nperseg=1250, noverlap=625, fs=1250)
    # ax.pcolorfast(t, f, sxx, cmap="YlGn", vmax=0.05)
    # ax.set_ylabel("Frequency (Hz)")
    # ax.set_xlabel("Time (s)")
    # # plt.pcolormesh(bispec_freq, bispec_freq, bispec, vmin=0, vmax=0.1, cmap="YlGn")
    # ax.set_ylim([1, 75])

fig.suptitle("fourier and bicoherence analysis of strong theta during MAZE")
# endregion

