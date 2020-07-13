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
import scipy.ndimage as smooth

basePath = [
    "/data/Clustering/SleepDeprivation/RatJ/Day1/",
    "/data/Clustering/SleepDeprivation/RatK/Day1/",
    "/data/Clustering/SleepDeprivation/RatN/Day1/",
    # "/data/Clustering/SleepDeprivation/RatJ/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day2/",
    # "/data/Clustering/SleepDeprivation/RatN/Day2/",
]


sessions = [processData(_) for _ in basePath]

plt.clf()
fig = plt.figure(1, figsize=(10, 15))
gs = gridspec.GridSpec(4, 4, figure=fig)
fig.subplots_adjust(hspace=0.3, wspace=0.3)
fig.suptitle("Localsleep")
titlesize = 8


#%% Localsleep example plots
# region

for sub, sess in enumerate([sessions[2]]):

    sess.trange = np.array([])
    lfpSrate = sess.recinfo.lfpSrate
    lfp, _, _ = sess.spindle.best_chan_lfp()
    t = np.linspace(0, len(lfp) / lfpSrate, len(lfp))
    spikes = sess.spikes.times

    post = sess.epochs.post
    period = post
    period_duration = np.diff(period)
    spikes_sd = [
        cell[np.where((cell > period[0]) & (cell < period[1]))[0]] for cell in spikes
    ]
    frate = np.asarray([len(cell) / period_duration for cell in spikes_sd]).squeeze()
    sort_frate_indices = np.argsort(frate)
    spikes = [spikes[indx] for indx in sort_frate_indices]

    selectedEvents = sess.localsleep.events.sample(n=4)
    instfiring = sess.localsleep.instfiring
    t_instfiring = np.linspace(
        sess.localsleep.period[0], sess.localsleep.period[1], len(instfiring)
    )

    taround = 2
    for ind, period in enumerate(selectedEvents.itertuples()):

        ax = fig.add_subplot(gs[0, ind])
        lfp_period = lfp[(t > period.start - taround) & (t < period.end + taround)]
        t_period = np.linspace(
            period.start - taround, period.end + taround, len(lfp_period)
        )
        instfiring_period = instfiring[
            (t_instfiring > period.start - taround)
            & (t_instfiring < period.end + taround)
        ]
        inst_tperiod = t_instfiring[
            (t_instfiring > period.start - taround)
            & (t_instfiring < period.end + taround)
        ]

        # ax.plot([period.start, period.start], [0, 100], "r")
        # ax.plot([period.end, period.end], [0, 100], "k")
        ax.fill_between(
            [period.start, period.end], [0, 0], [90, 90], alpha=0.3, color="#BDBDBD"
        )
        ax.fill_between(
            inst_tperiod, instfiring_period / 50, alpha=0.3, color="#212121",
        )
        ax.plot(
            t_period,
            stats.zscore(lfp_period) * 4 + len(spikes) + 15,
            "k",
            linewidth=0.8,
        )

        cmap = mpl.cm.get_cmap("inferno_r")

        for cell, spk in enumerate(spikes):
            color = cmap(cell / len(spikes))

            spk = spk[(spk > period.start - taround) & (spk < period.end + taround)]
            ax.plot(spk, cell * np.ones(len(spk)), "|", color=color, markersize=2)

        ax.set_title(f"{round(period.duration,2)} s")
        ax.axis("off")


# endregion


#%% Instantenous firing around localsleep
# region
colors = ["#ff928a", "#424242", "#3bceac"]
ax = fig.add_subplot(gs[1, 0])
ax.clear()

for sub, sess in enumerate(sessions):
    sess.trange = np.array([])

    tstart = sess.epochs.post[0]
    tend = sess.epochs.post[0] + 5 * 3600

    fbefore = sess.localsleep.instfiringbefore[:-1].mean(axis=0)
    fbeforestd = sess.localsleep.instfiringbefore[:-1].std(axis=0) / np.sqrt(
        len(sess.localsleep.events)
    )
    fafter = sess.localsleep.instfiringafter[:-1].mean(axis=0)
    fafterstd = sess.localsleep.instfiringafter[:-1].std(axis=0) / np.sqrt(
        len(sess.localsleep.events)
    )
    tbefore = np.linspace(-1, 0, len(fbefore))
    tafter = np.linspace(0.2, 1.2, len(fafter))

    # ax.fill_between(
    #     [0, 0.2],
    #     [min(fbefore), min(fbefore)],
    #     [max(fbefore), max(fbefore)],
    #     color="#BDBDBD",
    #     alpha=0.3,
    # )
    ax.fill_between(
        tbefore, fbefore + fbeforestd, fbefore - fbeforestd, color="#BDBDBD"
    )
    # ax.plot(tbefore, fbefore, color="#616161")
    ax.fill_between(tafter, fafter + fafterstd, fafter - fafterstd, color="#BDBDBD")
    # ax.plot(tafter, fafter, color="#616161")

    # self.events["duration"].plot.kde(ax=ax, color="k")
    # ax.set_xlim([0, max(self.events.duration)])
    ax.set_xlabel("Time from local sleep (s)")
    ax.set_ylabel("Instantneous firing")
    ax.set_xticks([-1, -0.5, 0, 0.2, 0.7, 1.2])
    ax.set_xticklabels(["-1", "-0.5", "start", "end", "0.5", "1"], rotation=45)


# ax = fig.add_subplot(3, 5, 13)

# for sub, sess in enumerate(sessions):
#     tstart = sess.epochs.post[0]
#     tend = sess.epochs.post[0] + 5 * 3600
#     sess.localsleep.events.duration.plot.kde(color="#BDBDBD")
# endregion


#%% localsleep and ripples around it
# region
colors = ["#ff928a", "#424242", "#3bceac"]
ax = fig.add_subplot(gs[1, 1])
ax.clear()
for sub, sess in enumerate(sessions):

    sess.trange = np.array([])
    locsleep = sess.localsleep.events
    ripples = sess.ripple.time

    tbin_before = lambda t: np.linspace(t - 1, t, 20)
    tbin_after = lambda t: np.linspace(t, t + 1, 20)
    ripples_counts_before = np.asarray(
        [
            np.histogram(ripples[:, 0], bins=tbin_before(event))[0]
            for event in locsleep.start
        ]
    )
    ripples_counts_after = np.asarray(
        [
            np.histogram(ripples[:, 0], bins=tbin_after(event))[0]
            for event in locsleep.end
        ]
    )

    total_ripples_before = np.sum(ripples_counts_before, axis=0)
    total_ripples_after = np.sum(ripples_counts_after, axis=0)

    combined = np.concatenate((total_ripples_before, total_ripples_after))

    subname = sess.sessinfo.session.sessionName

    ax.plot(
        np.linspace(-1, 0, 19),
        total_ripples_before,
        color=colors[sub],
        label=subname,
        lw=2,
        alpha=0.8,
    )
    ax.plot(
        np.linspace(0.5, 1.5, 19),
        total_ripples_after,
        color=colors[sub],
        lw=2,
        alpha=0.8,
    )
    ax.set_xlabel("Time from localsleep (s)")
    ax.set_ylabel("# SWRs")
ax.set_xticks([-1, -0.5, 0, 0.5, 1, 1.5])
ax.set_xticklabels([-1, -0.5, "start", "end", 0.5, 1])
ax.legend()
# endregion

#%% Numbers per minute during SD
# region
col = ["#FF8F00", "#388E3C", "#9C27B0"]

sd1 = np.zeros(3)
sd5 = np.zeros(3)
ax = fig.add_subplot(gs[1, 2])
ax.clear()
for sub, sess in enumerate(sessions):
    tstart = sess.epochs.post[0]
    tend = sess.epochs.post[0] + 5 * 3600

    tbin_offperiods = np.linspace(tstart, tend, 6)
    t_offperiods = sess.localsleep.events.start.values
    hist_off = np.histogram(t_offperiods, bins=tbin_offperiods)[0]
    hist_off = hist_off / 60
    sd1[sub] = hist_off[0]
    sd5[sub] = hist_off[-1]
    # plt.plot(hist_off / 60)

colsub = "#9E9E9E"
ax.plot(np.ones(3), sd1, "o", color=colsub, zorder=1)
ax.plot(3 * np.ones(3), sd5, "o", color=colsub, zorder=1)
ax.plot([1, 3], np.vstack((sd1, sd5)), color=colsub, linewidth=0.8, zorder=1)

mean_grp = np.array([np.mean(sd1), np.mean(sd5)])
sem_grp = np.array([stats.sem(sd1), stats.sem(sd5)])

ax.errorbar(
    np.array([1, 3]), mean_grp, yerr=sem_grp, color="#263238", fmt="o", zorder=2
)
# ax.plot([1, 3], [np.mean(sd1), np.mean(sd5)], color="#263238")
ax.set_xlim([0, 4])
ax.set_ylim([5, 25])
ax.set_xticks([1, 3])
ax.set_xticklabels(["SD1", "SD5"])
ax.set_ylabel("Numbers per min")

# endregion

#%% Spectrogram around localsleep

# region

for sub, sess in enumerate([sessions[0]]):

    sess.trange = np.array([])
    locslp = sess.localsleep.events
    eegSrate = sess.recinfo.lfpSrate
    nShanks = sess.recinfo.nShanks
    changrp = sess.recinfo.channelgroups[3]
    lfp = np.asarray(sess.utils.geteeg(chans=changrp[-1]))

    # lfp, chan, _ = sess.spindle.best_chan_lfp()
    # print(chan)
    t = np.linspace(0, len(lfp) / eegSrate, len(lfp))

    lfp_locslp_ind = []
    for evt in locslp.itertuples():
        lfp_locslp_ind.extend(
            np.arange(
                int((evt.start - 0.1) * eegSrate), int((evt.start + 0.1) * eegSrate)
            )
        )
    lfp_locslp_ind = np.asarray(lfp_locslp_ind)
    lfp_locslp = lfp[lfp_locslp_ind]
    lfp_locslp_avg = np.reshape(lfp_locslp, (len(locslp), 250)).mean(axis=0)
    t_locslp = np.linspace(0, len(lfp_locslp) / eegSrate, len(lfp_locslp))

    freqs = np.arange(20, 100, 0.5)
    wavdec = signal_process.wavelet_decomp(lfp_locslp, freqs=freqs)
    # wav = wavdec.cohen(ncycles=ncycles)
    wav = wavdec.cohen(ncycles=3)
    wav = (
        stats.zscore(wav, axis=1).reshape((wav.shape[0], 250, len(locslp))).mean(axis=2)
    )
    wav = smooth.gaussian_filter(wav, sigma=2)

    ax = fig.add_subplot(gs[2, 0])
    ax.pcolorfast(np.linspace(-100, 100, 250), freqs, wav, cmap="jet")
    ax2 = ax.twinx()
    ax2.plot(np.linspace(-100, 100, 250), lfp_locslp_avg, color="white")
    ax.spines["right"].set_visible(True)
    ax.set_xlabel("Time from start of localsleep periods (ms)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlim([-100, 100])

    lfp_locslp_ind = []
    for evt in locslp.itertuples():
        lfp_locslp_ind.extend(
            np.arange(int((evt.end - 0.1) * eegSrate), int((evt.end + 0.1) * eegSrate))
        )
    lfp_locslp_ind = np.asarray(lfp_locslp_ind)
    lfp_locslp = lfp[lfp_locslp_ind]
    lfp_locslp_avg = np.reshape(lfp_locslp, (len(locslp), 250)).mean(axis=0)
    t_locslp = np.linspace(0, len(lfp_locslp) / eegSrate, len(lfp_locslp))

    freqs = np.arange(20, 100, 0.5)
    wavdec = signal_process.wavelet_decomp(lfp_locslp, freqs=freqs)
    # wav = wavdec.cohen(ncycles=ncycles)
    wav = wavdec.cohen(ncycles=3)
    wav = (
        stats.zscore(wav, axis=1).reshape((wav.shape[0], 250, len(locslp))).mean(axis=2)
    )
    wav = smooth.gaussian_filter(wav, sigma=2)

    ax = fig.add_subplot(gs[2, 1])
    ax.pcolorfast(np.linspace(-100, 100, 250), freqs, wav, cmap="jet")
    ax2 = ax.twinx()
    ax2.plot(np.linspace(-100, 100, 250), lfp_locslp_avg, color="white")
    ax.spines["right"].set_visible(True)
    ax.set_xlabel("Time from end of localsleep (ms)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlim([-100, 100])


# endregion
