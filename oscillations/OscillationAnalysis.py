#%%
import warnings

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.pyplot import grid
import numpy as np
import pandas as pd
import scipy.signal as sg
import scipy.stats as stats
import seaborn as sns
from scipy.ndimage import gaussian_filter

import signal_process
from callfunc import processData
from ccg import correlograms
from plotUtil import Colormap, Fig

cmap = matplotlib.cm.get_cmap("hot_r")
warnings.simplefilter(action="default")


#%% Subjects
basePath = [
    "/data/Clustering/SleepDeprivation/RatJ/Day1/",
    "/data/Clustering/SleepDeprivation/RatK/Day1/",
    "/data/Clustering/SleepDeprivation/RatN/Day1/",
    "/data/Clustering/SleepDeprivation/RatJ/Day2/",
    "/data/Clustering/SleepDeprivation/RatK/Day2/",
    "/data/Clustering/SleepDeprivation/RatN/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day4/"
]


sessions = [processData(_) for _ in basePath]


#%% Bicoherence analysis on ripples
# region

colmap = Colormap().dynamicMap()

for sub, sess in enumerate(sessions):

    sess.trange = np.array([])
    eegSrate = sess.recinfo.lfpSrate
    ripples = sess.ripple.time
    lfp = sess.spindle.best_chan_lfp()[0]

    lfpripple = []
    for ripple in ripples:
        start = int(ripple[0] * eegSrate)
        end = int(ripple[1] * eegSrate)
        lfpripple.extend(lfp[start:end])

    lfpripple = np.asarray(lfpripple)
    bicoh, freq, bispec = signal_process.bicoherence(lfpripple, fhigh=300)

    bicohsmth = gaussian_filter(bicoh, sigma=3)
    # bicoh = np.where(bicoh > 0.05, bicoh, 0)
    plt.clf()
    fig = plt.figure(1, figsize=(10, 15))
    gs = gridspec.GridSpec(1, 2, figure=fig)
    fig.subplots_adjust(hspace=0.3)
    ax = fig.add_subplot(gs[0, 1])
    ax.clear()
    im = ax.pcolorfast(
        freq, freq, np.sqrt(bicohsmth), cmap=colmap, vmax=0.5, vmin=0.018
    )
    # ax.contour(freq, freq, bicoh, levels=[0.1, 0.2, 0.3], colors="k", linewidths=1)
    ax.set_ylim([1, max(freq) / 2])
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Frequency (Hz)")
    # cax = fig.add_axes([0.3, 0.8, 0.5, 0.05])
    # cax.clear()

    # ax.contour(freq, freq, bicoh, levels=[0.1, 0.2, 0.3], colors="k", linewidths=1)
    fig.colorbar(im, ax=ax, orientation="horizontal")

# endregion

#%% Gamma across SD
# region
group = []
plt.clf()
fig = plt.figure(1, figsize=(10, 15))
gs = gridspec.GridSpec(1, 3, figure=fig)
fig.subplots_adjust(hspace=0.3)
for sub, sess in enumerate(sessions[:3]):

    sess.trange = np.array([])
    lfp = sess.spindle.best_chan_lfp()[0]
    post = sess.epochs.post
    sd_period = [post[0], post[0] + 5 * 3600]
    t = np.linspace(0, len(lfp) / eegSrate, len(lfp))

    lfpsd = lfp[(t > sd_period[0]) & (t < sd_period[1])]
    tsd = np.linspace(sd_period[0], sd_period[1], len(lfpsd))
    binsd = np.linspace(sd_period[0], sd_period[1], 6)

    specgram = signal_process.spectrogramBands(lfpsd)

    gamma = stats.zscore(specgram.gamma)
    first_hour_gamma = gamma[(specgram.time > 0) & (specgram.time < 3600)]
    last_hour_gamma = gamma[(specgram.time > 4 * 3600) & (specgram.time < 5 * 3600)]

    hist_first, _ = np.histogram(first_hour_gamma, bins=np.arange(-3, 3, 0.1))
    hist_last, edge = np.histogram(last_hour_gamma, bins=np.arange(-3, 3, 0.1))

    ax = fig.add_subplot(gs[sub])
    ax.plot(edge[:-1], hist_first, "k", label="1st hour")
    ax.plot(edge[:-1], hist_last, "r", label="5th hour")
    ax.set_yscale("log")
    ax.set_xlabel("zscored values")
    ax.set_ylabel("Counts")
    ax.legend()

# endregion

#%% PSD during sleep deprivation from first to last hour to observe gamma reduction
# region
plt.clf()
fig = plt.figure(1, figsize=(10, 15))
gs = gridspec.GridSpec(1, 3, figure=fig)
fig.subplots_adjust(hspace=0.3)

for sub, sess in enumerate(sessions):
    sess.trange = np.array([])
    eegSrate = sess.recinfo.lfpSrate
    channel = sess.theta.bestchan
    post = sess.epochs.post
    eeg = sess.utils.geteeg(chans=channel, timeRange=[post[0], post[0] + 5 * 3600])
    nfrms_hour = eegSrate * 3600
    lfp1st = eeg[:nfrms_hour]
    lfp5th = eeg[-nfrms_hour:]

    psd = lambda sig: sg.welch(
        sig, fs=eegSrate, nperseg=10 * eegSate, noverlap=5 * eegSrate
    )
    multitaper = lambda sig: signal_process.mtspect(
        sig, fs=eegSrate, nperseg=10 * eegSrate, noverlap=5 * eegSrate
    )

    _, psd1st = multitaper(lfp1st)
    f, psd5th = multitaper(lfp5th)

    ax = fig.add_subplot(gs[sub])
    ax.loglog(f, psd1st, "k", label="ZT1")
    ax.loglog(f, psd5th, "r", label="ZT5")
    ax.set_xlim([1, 120])
    ax.set_ylim([10, 10e5])
    ax.set_title(sess.sessinfo.session.sessionName)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (A.U.)")
    ax.legend()


# endregion

#%% Plot spectrogram for different frequency bands (gamma, ripple) separately
# region
plt.clf()
fig = plt.figure(1, figsize=(10, 15))
gs = gridspec.GridSpec(2, 3, figure=fig)
fig.subplots_adjust(hspace=0.3)
for sub, sess in enumerate(sessions[:3]):
    sess.trange = np.array([])
    post = sess.epochs.post
    sd_period = [post[0], post[0] + 5 * 3600]
    thetachan = sess.theta.bestchan
    lfpsd = sess.utils.geteeg(chans=thetachan, timeRange=sd_period)
    specgram = signal_process.spectrogramBands(lfpsd, window=4, overlap=2)
    sxx = specgram.sxx
    freq = specgram.freq
    gamma = sxx[np.where((freq > 30) & (freq < 90))[0], :]
    ripple = sxx[np.where((freq > 150) & (freq < 250))[0], :]

    ax = fig.add_subplot(gs[0, sub])
    ax.imshow(gamma, aspect="auto")

    ax = fig.add_subplot(gs[1, sub])
    ax.imshow(ripple, aspect="auto")


# endregion

#%% Ripple rate in recovery sleep compared to last hour of sleep deprivation
# region
plt.clf()
fig = plt.figure(1, figsize=(10, 15))
gs = gridspec.GridSpec(1, 2, figure=fig)
fig.subplots_adjust(hspace=0.3)

rplrt, nrem_dur = [], []
for sub, sess in enumerate(sessions):
    sess.trange = np.array([])
    states = sess.brainstates.states
    ripples = sess.ripple.time
    post = sess.epochs.post
    states_recvslp = states.loc[
        (states.name == "nrem") & (states.start > post[0] + 5 * 3600)
    ].copy()
    nrem_bins = np.asarray(states_recvslp[["start", "end"]]).ravel()
    nrpls = np.histogram(ripples, bins=nrem_bins)[0][::2]
    rpl_rate = nrpls / np.diff(nrem_bins)[::2]

    nrem_dur.append(np.sum(np.diff(nrem_bins)[::2]))
    rplrt.append(rpl_rate)


ax = fig.add_subplot(gs[0])

for i in range(3):
    rplrt_sd_mean = [np.mean(rplrt[_]) for _ in [i, i + 3]]
    rplrt_sd_std = [np.std(rplrt[_]) / np.sqrt(len(rplrt[_])) for _ in [i, i + 3]]
    ax.errorbar([1, 2], rplrt_sd_mean, yerr=rplrt_sd_std, color="gray")


ax.set_xticks([1, 2])
ax.set_xticklabels(["SD", "NSD"])
ax.set_xlim([0, 3])
ax.set_ylim([0.3, 1.2])
ax.set_ylabel("Ripple rate (Hz)")
ax.set_title("Mean ripple rate compared between \n recovery sleep and control session")


ax = fig.add_subplot(gs[1])
ax.bar([1, 2], [np.mean(nrem_dur[:3]), np.mean(nrem_dur[3:])], color="gray", width=0.6)
ax.set_xticks([1, 2])
ax.set_xticklabels(["SD", "NSD"])
ax.set_xlim([0, 3])
ax.set_ylabel("Duration (s)")
ax.set_title("Total NREM duration")
# ax.errorbar([1, 2], [np.mean(nrem_dur[:3]), np.mean(nrem_dur[3:])])


# endregion

#%% Hswa-Ripple locking
# region
psth_all = []
for sub, sess in enumerate(sessions[:3]):
    sess.trange = np.array([])
    post = sess.epochs.post
    period = [post[0] + 5 * 3600, post[1]]
    psth = sess.eventpsth.hswa_ripple.compute(period=None)
    psth_all.append(psth)
psth_all = np.dstack(psth_all)
psth_ = np.mean(psth_all, axis=2)
plt.plot(psth_.T)

# endregion

#%% Mean hswa amplitude plotted over sleep-wake cycle
# region
plt.clf()
fig = plt.figure(1, figsize=(10, 15))
gs = gridspec.GridSpec(6, 1, figure=fig)
fig.subplots_adjust(hspace=0.3)
for sub, sess in enumerate(sessions):
    sess.trange = np.array([])
    pre = sess.epochs.pre
    post = sess.epochs.post
    swa = sess.swa.time
    peakamp = sess.swa.peakamp
    bins = np.arange(pre[0], post[1], 60)
    mean_swa_amp = stats.binned_statistic(swa, peakamp, statistic="mean", bins=bins)

    ax = fig.add_subplot(gs[sub])
    ax.plot(mean_swa_amp.statistic)

# endregion

#%% wavelet analysis around delta oscillations
# region
for sub, sess in enumerate(sessions):
    sess.trange = np.array([])


# endregion

#%% Plot spectrogram for maze periods (checking novelty induced high theta)
# region
plt.clf()
fig = plt.figure(1, figsize=(10, 15))
gs = gridspec.GridSpec(6, 1, figure=fig)
fig.subplots_adjust(hspace=0.4)

for sub, sess in enumerate(sessions):
    sess.trange = np.array([])
    eegSrate = sess.recinfo.lfpSrate
    maze = sess.epochs.maze
    lfp = sess.utils.geteeg(chans=sess.theta.bestchan, timeRange=maze)
    specgram = signal_process.spectrogramBands(
        lfp, sampfreq=eegSrate, window=4, overlap=2
    )
    ax = fig.add_subplot(gs[sub])
    specgram.plotSpect(ax=ax, freqRange=[1, 30])
    # ax.plot(specgram.theta_delta_ratio)


# endregion

#%% Hourly coupling of delta and ripples over POST
# region
plt.clf()
fig = plt.figure(1, figsize=(10, 15))
gs = gridspec.GridSpec(6, 1, figure=fig)
fig.subplots_adjust(hspace=0.4)

for sub, sess in enumerate(sessions):
    sess.trange = np.array([])
    eegSrate = sess.recinfo.lfpSrate

    post = sess.epochs.post
    windows = np.arange(post[0], post[1], 2 * 3600)

    ax = fig.add_subplot(gs[sub])
    for start in windows[:-1]:
        period = [start, start + 2 * 3600]
        psth = sess.eventpsth.hswa_ripple.compute(period=period, nQuantiles=1)
        ax.plot(psth.squeeze())

# endregion

#%%* Ripple and delta waves density during Sleep deprivation
# region
figure = Fig()
fig, gs = figure.draw(grid=[5, 5])
axripple = plt.subplot(gs[0, 0])
figure.panel_label(axripple, "a")
axdelta = plt.subplot(gs[1, 0])

data = []
for sub, sess in enumerate(sessions[:3]):
    sess.trange = np.array([])
    post = sess.epochs.post
    rpls = sess.ripple.time
    swa = sess.swa.time

    sd = np.linspace(post[0], post[0] + 5 * 3600, 6)
    rpls_sd = np.histogram(rpls, bins=sd)[0] / 3600
    swa_sd = np.histogram(swa, bins=sd)[0] / 60

    window = [f"ZT-{_}" for _ in range(len(sd) - 1)]
    data.append(pd.DataFrame({"hour": window, "ripple": rpls_sd, "delta": swa_sd}))

density = pd.concat(data)
sns.barplot(x="hour", y="ripple", data=density, ci="sd", ax=axripple)
axripple.set_ylabel("Ripples/s")
axripple.tick_params(axis="x", labelrotation=45)

sns.barplot(x="hour", y="delta", data=density, ci="sd")
axdelta.set_ylabel("delta/min")
axdelta.set_xlabel("")
axdelta.tick_params(axis="x", labelrotation=45)

figure.savefig("ripple_delta_density_sd", __file__)
# endregion

#%%* Ripple, delta and spindle density during recovery sleep
# region
figure = Fig()
fig, gs = figure.draw(grid=[5, 5])

data = []
for sub, sess in enumerate(sessions[:3]):
    sess.trange = np.array([])
    post = sess.epochs.post
    rpls = sess.ripple.time
    swa = sess.swa.time
    spndl = sess.spindle.time

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
    swa_nrem = np.histogram(swa, bins=nrem_bins)[0][::2] / nrem_dur
    spndl_nrem = np.histogram(spndl, bins=nrem_bins)[0][::2] / nrem_dur

    window = [nrem_ind + 1 for nrem_ind in range(len(nrem_dur))]
    data.append(
        pd.DataFrame(
            {
                "nrem": window,
                "ripple": rpls_nrem,
                "delta": swa_nrem,
                "spindle": spndl_nrem,
            }
        )
    )

density = pd.concat(data)
# order = np.unique(density.nrem)

axripple = plt.subplot(gs[0, 1:3])
axripple.clear()
sns.barplot(
    x="nrem", y="ripple", data=density, ci="sd", ax=axripple,
)
axripple.set_ylabel("Ripples / s")
axripple.tick_params(axis="x", labelrotation=45)
figure.panel_label(axripple, "b")

axdelta = plt.subplot(gs[1, 1:3])
sns.barplot(x="nrem", y="delta", data=density, ci="sd")
axdelta.set_ylabel("delta / s")
axdelta.tick_params(axis="x", labelrotation=45)

axspndl = plt.subplot(gs[0, 3:])
sns.barplot(x="nrem", y="spindle", data=density, ci="sd")
axspndl.set_ylabel("spindle / s")
axspndl.tick_params(axis="x", labelrotation=45)

figure.savefig("ripple_delta_spindle_recoverysleep", __file__)
# endregion

#%%* Hourly delta ripple coupling over SD and comparing with chance level
# region

psth_all = []
psthchance_all = []
for sub, sess in enumerate(sessions[:3]):
    sess.trange = np.array([])
    eegSrate = sess.recinfo.lfpSrate
    rpls = sess.ripple.time[:, 0]
    swa = sess.swa.time
    post = sess.epochs.post
    # windows = np.arange(post[0], post[0] + 5 * 3600, 3600)
    windows = [post[0], post[0] + 2 * 3600]
    npsth = len(windows)

    psth, psthchance = [], []

    for start in windows:
        period = [start, start + 2 * 3600]
        nrpls_window = len(rpls[(rpls > period[0]) & (rpls < period[1])])
        swa_window = swa[(swa > period[0]) & (swa < period[1])]
        rand_events = (
            np.diff(period)[0] * np.random.random_sample((nrpls_window,)) + period[0]
        )
        psth.append(
            sess.eventpsth.hswa_ripple.compute(period=period, nQuantiles=1).squeeze()
        )
        psthchance.append(sess.eventpsth.compute(swa, rand_events).squeeze())

    psth = np.asarray(psth)
    assert psth.shape[0] == len(windows)
    psth = pd.DataFrame(psth.T, columns=np.arange(1, npsth + 1))
    psth["time"] = np.linspace(-0.5, 0.5, 101) + 0.05
    psth = pd.melt(
        psth,
        id_vars=["time"],
        value_vars=np.arange(1, npsth + 1),
        var_name="window",
        value_name="counts",
    )
    psth["subname"] = [sess.sessinfo.session.sessionName] * len(psth)
    psth_all.append(psth)

    psthchance = np.asarray(psthchance)
    psthchance = pd.DataFrame(psthchance.T, columns=np.arange(1, npsth + 1))
    psthchance["time"] = np.linspace(-0.5, 0.5, 101) + 0.05
    psthchance = pd.melt(
        psthchance,
        id_vars=["time"],
        value_vars=np.arange(1, npsth + 1),
        var_name="window",
        value_name="counts",
    )
    psthchance["subname"] = [sess.sessinfo.session.sessionName] * len(psth)
    psthchance_all.append(psthchance)


psth_all = pd.concat(psth_all, ignore_index=True)
psthchance_all = pd.concat(psthchance_all, ignore_index=True)
# TODO chance level calculation
figure = Fig()
fig, gs = figure.draw(grid=[5, 5])
axdel_rpl = fig.add_subplot(gs[2, 0])
sns.lineplot(
    x="time",
    y="counts",
    hue="window",
    data=psth_all,
    palette="copper_r",
    ci=None,
    # n_boot=10,
    # seed=10,
    legend=False,
    ax=axdel_rpl,
)
# sns.lineplot(
#     x="time",
#     y="counts",
#     hue="window",
#     data=psthchance_all,
#     # palette="copper_r",
#     ci=48,
#     n_boot=10,
#     seed=10,
#     legend=False,
#     ax=axdel_rpl,
# )
axdel_rpl.set_xlabel("Times from delta (s)")
axdel_rpl.set_ylabel("Counts")
axdel_rpl.set_title("Ripple probability over SD")

figure.savefig("delta_ripple_coupling_sd", __file__)
# endregion

#%%* Hourly delta-ripple locking over the course of normal sleep (control sessions POST)
# region
psth_all = []
for sub, sess in enumerate(sessions[3:]):
    sess.trange = np.array([])
    eegSrate = sess.recinfo.lfpSrate

    post = sess.epochs.post
    windows = np.arange(post[0], post[1], 2 * 3600)

    psth = []

    for start in windows[:5]:
        period = [start, start + 2 * 3600]
        psth.append(
            sess.eventpsth.hswa_ripple.compute(period=period, nQuantiles=1).squeeze()
        )

    psth = np.asarray(psth)
    psth = pd.DataFrame(psth.T, columns=np.arange(1, 6))
    psth["time"] = np.linspace(-0.5, 0.5, 101) + 0.05
    psth = pd.melt(
        psth,
        id_vars=["time"],
        value_vars=np.arange(1, 6),
        var_name="window",
        value_name="counts",
    )
    psth["subname"] = [sess.sessinfo.session.sessionName] * len(psth)

    psth_all.append(psth)

figure = Fig()
fig, gs = figure.draw(grid=[5, 5])
psth_all = pd.concat(psth_all, ignore_index=True)
ax = fig.add_subplot(gs[3])
ax.clear()
sns.lineplot(
    x="time",
    y="counts",
    hue="window",
    data=psth_all,
    palette="copper_r",
    ci=None,
    # n_boot=1,
    legend=False,
)
ax.set_xlabel("Times from delta (s)")
ax.set_ylabel("Counts")
ax.set_title("Ripple probability \n over normal sleep")
figure.panel_label(ax, "e")
figure.savefig("normal_sleep_hourly_delta_ripple", __file__)
# endregion


#%%* Ripple probability vs delta amplitude divided into quantiles
# region
psth_all = []
for sub, sess in enumerate(sessions):
    sess.trange = np.array([])
    post = sess.epochs.post
    period = [post[0] + 5 * 3600, post[1]]
    psth = sess.eventpsth.hswa_ripple.compute(period=None)
    psth = pd.DataFrame(psth.T, columns=range(10))
    psth["time"] = np.linspace(-0.5, 0.5, 101) + 0.05
    psth = pd.melt(
        psth,
        id_vars=["time"],
        value_vars=range(10),
        var_name="quantile",
        value_name="counts",
    )
    psth["subname"] = [sess.sessinfo.session.sessionName] * len(psth)

    psth_all.append(psth)

psth_all = pd.concat(psth_all, ignore_index=True)

figure = Fig()
fig, gs = figure.draw(grid=[5, 5])
ax = fig.add_subplot(gs[3, 1])
sns.lineplot(
    x="time",
    y="counts",
    hue="quantile",
    data=psth_all,
    palette="Greens",
    ci=None,
    # n_boot=1,
    legend=False,
)
ax.set_xlabel("Times from delta (s)")
ax.set_ylabel("Counts")
ax.set_title("Ripple probability")
figure.panel_label(ax, "d")
figure.savefig("ripple_delta_probability", __file__)
# endregion

#%%* hswa - spindle coactivity
# region
psth_all = []
for sub, sess in enumerate(sessions):
    sess.trange = np.array([])
    post = sess.epochs.post
    period = [post[0] + 5 * 3600, post[1]]
    psth = sess.eventpsth.hswa_spindle.compute(period=None, nQuantiles=10)
    psth = pd.DataFrame(psth.T, columns=range(10))
    psth["time"] = np.linspace(-0.5, 0.5, 101) + 0.05
    psth = pd.melt(
        psth,
        id_vars=["time"],
        value_vars=range(10),
        var_name="quantile",
        value_name="counts",
    )
    psth["subname"] = [sess.sessinfo.session.sessionName] * len(psth)

    psth_all.append(psth)

psth_all = pd.concat(psth_all, ignore_index=True)

figure = Fig()
fig, gs = figure.draw(grid=[5, 5])
ax = fig.add_subplot(gs[1, 0])
sns.lineplot(
    x="time",
    y="counts",
    hue="quantile",
    data=psth_all,
    palette="Greens",
    ci=None,
    # n_boot=1,
    legend=False,
)
ax.set_xlabel("Times from delta (s)")
ax.set_ylabel("Counts")
ax.set_title("Spindle probability")
figure.panel_label(ax, "a")
figure.savefig("spindle_delta_coupling", __file__)
# endregion

#%% Hourly delta-ripple coupling over recovery sleep
# region
psth_all = []
for sub, sess in enumerate(sessions[:3]):
    sess.trange = np.array([])
    eegSrate = sess.recinfo.lfpSrate

    post = sess.epochs.post
    windows = np.arange(post[0] + 5 * 3600, post[1], 3600)

    psth = []

    for start in windows[:5]:
        period = [start, start + 3600]
        psth.append(
            sess.eventpsth.hswa_ripple.compute(period=period, nQuantiles=1).squeeze()
        )

    psth = np.asarray(psth)
    psth = pd.DataFrame(psth.T, columns=np.arange(1, 6))
    psth["time"] = np.linspace(-0.5, 0.5, 101) + 0.05
    psth = pd.melt(
        psth,
        id_vars=["time"],
        value_vars=np.arange(1, 6),
        var_name="window",
        value_name="counts",
    )
    psth["subname"] = [sess.sessinfo.session.sessionName] * len(psth)

    psth_all.append(psth)


psth_all = pd.concat(psth_all, ignore_index=True)
ax = fig.add_subplot(gs[2, 1])
sns.lineplot(
    x="time",
    y="counts",
    hue="window",
    data=psth_all,
    palette="copper_r",
    ci=None,
    # n_boot=1,
    legend=False,
)
ax.set_xlabel("Times from delta (s)")
ax.set_ylabel("Counts")
ax.set_title("Ripple probability \n during recovery sleep", fontsize=titlesize)
# endregion


#%% Ripple stats during sleep deprivation
# region
plt.clf()
group = []
for sub, sess in enumerate(sessions):

    sess.trange = np.array([])

    ripples = sess.ripple.time
    duration = np.diff(ripples, axis=1).squeeze()
    peakpower = sess.ripple.peakpower
    tstart = sess.epochs.post[0]
    tend = sess.epochs.post[0] + 5 * 3600
    nbin = 5
    tbin = np.linspace(tstart, tend, nbin + 1)
    colors = [cmap(_) for _ in np.linspace(0, 1, nbin)]
    nripples = []
    ripple_dur = []
    ripple_cat = []
    peakpower_all = []
    for i in range(nbin):
        binstart = tbin[i]
        binend = tbin[i + 1]

        ripple_ind = np.where((ripples[:, 0] > binstart) & (ripples[:, 0] < binend))[0]
        dur_bin = duration[ripple_ind]
        bin_cat = (i + 1) * np.ones(len(dur_bin))
        power_bin = peakpower[ripple_ind]
        # peakpowerbin = peakpower[ripple_ind]
        # powerbinning = np.logspace(np.log10(1.2), np.log10(60), 50)
        # powerbinning = np.linspace(5, 40,)
        # peakhist, _ = np.histogram(peakpowerbin, bins=powerbinning)
        # peakhist = peakhist / np.sum(peakhist)
        # plt.plot(powerbinning[:-1], peakhist, color=colors[i])
        # plt.yscale("log")
        nripples.append(len(ripple_ind))
        ripple_dur.extend(dur_bin)
        ripple_cat.extend(bin_cat)
        peakpower_all.extend(power_bin)

    nripples = np.array(nripples)
    data = pd.DataFrame(
        {"dur": ripple_dur, "hour": ripple_cat, "peakpower": peakpower_all}
    )
    subname = sess.sessinfo.session.name
    day = sess.sessinfo.session.day

    plt.subplot(3, 3, sub + 1)
    # plt.bar(np.arange(0.5, 5.5, 1), nripples)
    sns.countplot(x="hour", data=data, color="#f0b67f")
    plt.title(f"{subname} {day} SD")

    plt.subplot(3, 3, sub + 3 + 1)
    sns.violinplot(x="hour", y="dur", data=data, color="#4CAF50")
    # plt.plot(ripple_dur)
    plt.ylabel("duration")

    plt.subplot(3, 3, sub + 6 + 1)
    sns.violinplot(x="hour", y="peakpower", data=data, color="#CE93D8")
    # plt.plot(ripple_dur)
    plt.ylabel("duration")

plt.suptitle("Ripples during Sleep deprivation period")
# endregion

#%% Ripple distribution during sleep deprivation
# region
plt.clf()
group = []
for sub, sess in enumerate(sessions):

    sess.trange = np.array([])

    ripples = sess.ripple.time
    peakpower = sess.ripple.peakpower
    tstart = sess.epochs.post[0]
    tend = sess.epochs.post[0] + 5 * 3600
    nbin = 5
    tbin = np.linspace(tstart, tend, nbin + 1)
    colors = [cmap(_) for _ in np.linspace(0, 1, nbin)]
    for i in range(nbin):
        binstart = tbin[i]
        binend = tbin[i + 1]

        ripple_ind = np.where((ripples[:, 0] > binstart) & (ripples[:, 0] < binend))[0]
        peakpowerbin = peakpower[ripple_ind]
        # powerbinning = np.logspace(np.log10(1.2), np.log10(60), 50)
        powerbinning = np.linspace(5, 40,)
        peakhist, _ = np.histogram(peakpowerbin, bins=powerbinning)
        peakhist = peakhist / np.sum(peakhist)
        plt.subplot(1, 3, sub + 1)
        plt.plot(powerbinning[:-1], peakhist, color=colors[i])
        plt.ylabel("fraction")
        plt.xlabel("zscore value")

    subname = sess.sessinfo.session.name
    day = sess.sessinfo.session.day

    # plt.yscale("log")
    plt.title(f"{subname} {day} SD")

plt.suptitle("Ripple distribution during sleep deprivation")
# endregion

#%% Spindles and Ripples around REM sleep
# region
plt.clf()
tbin = lambda x: np.arange(x - 80, x + 80)
for sub, sess in enumerate(sessions):

    sess.trange = np.array([])
    peakrpl = sess.ripple.peaktime
    peakspndl = sess.spindle.peaktime
    # hswa = sess.swa.time
    states = sess.brainstates.states
    remstart = states[states["name"] == "rem"].start.values

    rpl_rem = [np.histogram(peakrpl, bins=tbin(_))[0] for _ in remstart]
    spndl_rem = [np.histogram(peakspndl, bins=tbin(_))[0] for _ in remstart]

    t = tbin(0)[:-1] + 0.5
    rpl_rem = np.asarray(rpl_rem).sum(axis=0)
    spndl_rem = np.asarray(spndl_rem).sum(axis=0)

    plt.subplot(2, 3, sub + 1)
    plt.plot(t, spndl_rem, "blue", linewidth=2)
    plt.plot(t, rpl_rem, "r")

    subname = sess.sessinfo.session.name
    day = sess.sessinfo.session.day
    if sub < 3:
        plt.title(f"{subname} {day} SD")
    else:
        plt.title(f"{subname} {day} NSD")


plt.subplot(2, 3, 4)
plt.ylabel("Counts")
plt.xlabel("Time from REM onset (sec)")
plt.legend(["spindles", "ripples"])

plt.suptitle("Ripple and Spindles aroung REM sleep")
# endregion

#%% Ripples around Spindles
# region
tbin = lambda x: np.linspace(x - 5, x + 5, 40)
plt.clf()
for sub, sess in enumerate(sessions):

    sess.trange = np.array([])
    peakrpl = sess.ripple.peaktime
    peakspndl = sess.spindle.peaktime
    peakspndl = sess.swa.time

    hist_rpl = [np.histogram(peakrpl, bins=tbin(_))[0] for _ in peakspndl]

    t = tbin(0)[:-1] + (np.diff(tbin(0)).mean()) / 2
    hist_rpl = np.asarray(hist_rpl).sum(axis=0)
    plt.subplot(2, 3, sub + 1)

    plt.plot(t, hist_rpl, "#2d3143")

    subname = sess.sessinfo.session.name
    day = sess.sessinfo.session.day
    if sub < 3:
        plt.title(f"{subname} {day} SD")
    else:
        plt.title(f"{subname} {day} NSD")


plt.subplot(2, 3, 4)
plt.ylabel("Counts", fontweight="bold")
plt.xlabel("Time from spindle peak onset (sec)", fontweight="bold")

plt.suptitle("Ripples around Spindles ")
# endregion

#%% Ripples-spindles-SWA
# region
tbin = lambda x: np.linspace(x - 2, x + 2, 40)
plt.clf()
for sub, sess in enumerate(sessions):

    sess.trange = np.array([])
    peakrpl = sess.ripple.peaktime
    peakspndl = sess.spindle.peaktime
    hswa = sess.swa.tend

    hist_rpl_hswa = [np.histogram(peakrpl, bins=tbin(_))[0] for _ in hswa]


t = tbin(0)[:-1]
hist_rpl_hswa = np.asarray(hist_rpl_hswa).sum(axis=0)
plt.plot(t, hist_rpl_hswa)
# endregion


#%% Ripples SWA wavelet
# region
plt.clf()
group = []
for sub, sess in enumerate(sessions):

    sess.trange = np.array([])
    ripples = sess.ripple.time
    a, chans, _ = sess.ripple.best_chan_lfp()


duration = np.diff(ripples, axis=1)
peakpower = sess.ripple.peakpower
longrpl = np.where(peakpower > 20)[0]
ripples = np.delete(ripples, longrpl, 0)

lfp = stats.zscore(a[1, :])

# with progressbar.ProgressBar(max_value=len(ripples)) as bar:
# frequency = np.logspace(np.log10(100), np.log10(250), 100)


nbins = 400
frequency = np.linspace(1, 300, nbins)

baseline = wavelet_decomp(lfp[0:2500], lowfreq=1, highfreq=400, nbins=nbins)

# frequency = np.asarray([round(_) for _ in frequency])
# wavedecomp = np.zeros((100, 2500))
timepoints = ripples[[1, 30, 45, 3600, 76], 0]
timepoints = sess.swa.time[5:10]
for i, rpl in enumerate(timepoints):
    start = int(rpl * 1250) - 1250
    end = int(rpl * 1250) + 1250
    signal = lfp[start:end]
    wavedecomp_rpl = wavelet_decomp(signal, lowfreq=1, highfreq=300, nbins=nbins)
    # wavedecomp = wavedecomp + wavedecomp_rpl
    plt.subplot(3, 5, i + 1)
    # bar.update(i)
    # wavedecomp = (wavedecomp_rpl - np.mean(baseline)) / np.std(baseline)
    wavedecomp = wavedecomp_rpl
    plt.pcolormesh(np.linspace(-1, 1, 2500), frequency, wavedecomp, cmap="hot")

    plt.subplot(3, 5, i + 5 + 1)
    plt.plot(np.linspace(-1, 1, 2500), filter_sig.filter_delta(signal), "#aa9d9d")
    plt.subplot(3, 5, i + 10 + 1)
    plt.plot(np.linspace(-1, 1, 2500), stats.zscore(signal), "k")

# plt.colorbar()
# plt.yscale("log")
# plt.yticks(np.arange(0, 100, 10), frequency[np.arange(0, 100, 10)])

# lfpripple = np.asarray(lfpripple)

# wavedecomp1 = wavelet_decomp(lfpripple, lowfreq=1, highfreq=6, nbins=10)
# wavedecomp2 = wavelet_decomp(lfpripple, lowfreq=150, highfreq=250, nbins=20)
# specgrm.append(np.abs(wavedecomp))
# plt.subplot(5, 2, i)
# endregion

#%% hswa-Ripple-Gamma Bandcoupling
# region
group = []
plt.clf()
fig = plt.figure(1, figsize=(1, 15))
gs = GridSpec(3, 1, figure=fig)
fig.subplots_adjust(hspace=0.5)

for sub, sess in enumerate(sessions[:3]):

    sess.trange = np.array([])
    tstart = sess.epochs.post[0]
    tend = sess.epochs.post[0] + 5 * 3600
    lfp, _, _ = sess.spindle.best_chan_lfp()

    t = np.linspace(0, len(lfp) / 1250, len(lfp))
    # lfp = lfp[(t > tstart) & (t < tend)]
    bands = spectrogramBands(lfp)
    time = bands.time
    gamma = stats.zscore(bands.gamma)
    theta = stats.zscore(bands.theta)
    th_del_ratio = stats.zscore(bands.theta_delta_ratio)

    binwind = np.linspace(tstart, tend, 100)

    binlfp = lambda x, t1, t2: x[(time > t1) & (time < t2)]

    cross_corr = []
    for i in range(len(binwind) - 1):
        theta_bin = binlfp(theta, binwind[i], binwind[i + 1])
        gamma_bin = binlfp(gamma, binwind[i], binwind[i + 1])

        a = np.correlate(theta_bin, gamma_bin, "full")
        # cross_corr[i, :] = a[:413]

        cross_corr.append([np.asarray(a)])

    cross_corr = np.asarray(cross_corr).T

    ax = fig.add_subplot(gs[sub, 0])
    ax.imshow(cross_corr, aspect="auto")
    # theta_gamma = pd.DataFrame({"theta": theta, "gamma": gamma})
    # a = grangercausalitytests(theta_gamma, maxlag=1)

    #
    # ax.plot(time, gamma, "r")
    # ax.plot(time, theta, "g")
    # ax.plot(time, th_del_ratiocr, "k")
# endregion


#%% hswa vs population burst (PBE) psth
# region
# TODO change channel for delta detection closer to cortical electrodes
plt.clf()
fig = plt.figure(1, figsize=(10, 15))
gs = gridspec.GridSpec(2, 3, figure=fig)
fig.subplots_adjust(hspace=0.3)
psth_all = []
for sub, sess in enumerate(sessions):
    sess.trange = np.array([])
    swa = sess.swa.time
    swa_amp = sess.swa.peakamp
    # sess.pbe.detect()

    pbe = sess.pbe.events
    psth = sess.eventpsth.compute(ref=swa, event=pbe.start, quantparam=swa_amp)

    ax = fig.add_subplot(gs[sub])
    sess.eventpsth.plot(ax=ax)

# endregion

