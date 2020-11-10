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
# warnings.simplefilter(action="default")


#%% Subjects
basePath = [
    # "/data/Clustering/SleepDeprivation/RatJ/Day1/",
    # "/data/Clustering/SleepDeprivation/RatK/Day1/",
    "/data/Clustering/SleepDeprivation/RatN/Day1/",
    # "/data/Clustering/SleepDeprivation/RatJ/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day2/",
    "/data/Clustering/SleepDeprivation/RatN/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day4/"
]


sessions = [processData(_) for _ in basePath]


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
        powerbinning = np.linspace(
            5,
            40,
        )
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

#%% Gamma oscillation power vs velocity/position of animal
# region

figure = Fig()
fig, gs = figure.draw(grid=[4, 2])
for sub, sess in enumerate(sessions):
    eegSrate = sess.recinfo.lfpSrate
    maze = sess.epochs.maze
    posdata = sess.position.data
    posdata = posdata[(posdata.time > maze[0]) & (posdata.time < maze[1])]
    chan = sess.theta.bestchan
    lfpmaze = sess.recinfo.geteeg(chans=chan, timeRange=maze)
    gamma_lfp = signal_process.filter_sig.bandpass(lfpmaze, lf=25, hf=100)
    gamma_amp = np.abs(signal_process.hilbertfast(gamma_lfp))
    gamma_t = np.linspace(maze[0], maze[1], len(gamma_lfp))

    peakgamma = sess.gamma.get_peak_intervals(
        lfpmaze, band=(25, 100), lowthresh=1, highthresh=2
    )
    peakgamma = peakgamma / eegSrate + maze[0]

    gamma_pow = stats.binned_statistic(
        gamma_t, gamma_amp, bins=np.concatenate(peakgamma)
    )[0][::2]

    gamma_pos = stats.binned_statistic(
        posdata.time,
        [posdata.x, posdata.y, posdata.time],
        bins=np.concatenate(peakgamma),
    )[0]

    x_gamma, y_gamma, t_gamma = gamma_pos[0][::2], gamma_pos[1][::2], gamma_pos[2][::2]

    # plt.plot(posdata.time, posdata.y)
    # plt.plot(t_gamma, y_gamma, "r.")

    x_grid = np.linspace(np.min(posdata.x), np.max(posdata.x), 100)
    y_grid = np.linspace(np.min(posdata.y), np.max(posdata.y), 100)

    hist2 = stats.binned_statistic_2d(
        x_gamma, y_gamma, gamma_pow, bins=[x_grid, y_grid]
    )
    hist_pow = hist2[0]
    hist_pow = np.nan_to_num(hist_pow, nan=0)

    ax = plt.subplot(gs[sub])
    ax.pcolormesh(
        hist2.x_edge,
        hist2.y_edge,
        gaussian_filter(hist_pow, sigma=2),
        cmap="jet",
        rasterized=True,
    )
    ax.axis("off")

# figure.savefig("gamma_power_track", __file__)
# endregion
