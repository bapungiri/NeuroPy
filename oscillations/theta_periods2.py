# %%

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as sg
import scipy.stats as stats
import seaborn as sns
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter, gaussian_filter1d
import ipywidgets as widgets
import random
from sklearn import linear_model
import statsmodels.api as sm
import pingouin as pg

from callfunc import processData
import signal_process
from mathutil import threshPeriods
import warnings

warnings.simplefilter(action="default")

#%% ====== functions needed for some computation ============
# region
def doWavelet(lfp, freqs, ncycles=3):
    wavdec = signal_process.wavelet_decomp(lfp, freqs=freqs)
    # wav = wavdec.cohen(ncycles=ncycles)
    wav = wavdec.colgin2009()

    wav = stats.zscore(wav)
    wav = gaussian_filter(wav, sigma=4)

    return wav


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

#%% Subjects to choose from
basePath = [
    # "/data/Clustering/SleepDeprivation/RatJ/Day1/",
    # "/data/Clustering/SleepDeprivation/RatK/Day1/",
    # "/data/Clustering/SleepDeprivation/RatN/Day1/",
    # "/data/Clustering/SleepDeprivation/RatJ/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day2/",
    "/data/Clustering/SleepDeprivation/RatN/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day4/"
]
sessions = [processData(_) for _ in basePath]

#%% Phase-amplitude comodulogram for multiple frequencies
# region

# during REM sleep
plt.clf()
fig = plt.figure(1, figsize=(1, 15))
gs = GridSpec(2, 3, figure=fig)
fig.subplots_adjust(hspace=0.5)

colband = ["#CE93D8", "#1565C0", "#E65100"]
p = Pac(idpac=(6, 3, 0), f_pha=(4, 10, 1, 1), f_amp=(30, 100, 5, 5))

for sub, sess in enumerate(sessions):

    sess.trange = np.array([])
    tstart = sess.epochs.post[0]
    tend = sess.epochs.post[0] + 5 * 3600
    lfp, _, _ = sess.spindle.best_chan_lfp()
    t = np.linspace(0, len(lfp) / 1250, len(lfp))
    states = sess.brainstates.states

    if sub < 3:
        plt_ind = sub
        # color = "r"
        # color = colband[sub]
        lnstyle = "solid"
        rem = states[(states["start"] > tend) & (states["name"] == "rem")]
    else:
        plt_ind = sub - 3
        # color = colband[sub - 3]
        lnstyle = "dashed"
        rem = states[(states["start"] > tstart) & (states["name"] == "rem")]

    binlfp = lambda x, t1, t2: x[(t > t1) & (t < t2)]
    freqIntervals = [[30, 50], [50, 90], [100, 150]]  # in Hz

    lfprem = []
    for epoch in rem.itertuples():
        lfprem.extend(binlfp(lfp, epoch.start, epoch.end))

    lfprem = np.asarray(lfprem)

    xpac = p.filterfit(1250.0, lfprem, n_perm=20)
    theta_lfp = stats.zscore(filter_sig.filter_theta(lfprem))
    hil_theta = hilbertfast(theta_lfp)
    theta_amp = np.abs(hil_theta)
    theta_angle = np.angle(hil_theta, deg=True) + 180
    angle_bin = np.arange(0, 360, 20)
    bin_ind = np.digitize(theta_angle, bins=angle_bin)

    ax = fig.add_subplot(gs[sub])
    # ax.plot(
    #     angle_bin[:-1] + 10, mean_amp_norm, linestyle=lnstyle, color=colband[band]
    # )
    # ax.set_xlabel("Degree (from theta trough)")
    # ax.set_ylabel("Amplitude")
    p.comodulogram(
        xpac.mean(-1),
        title="Contour plot with 5 regions",
        cmap="Spectral_r",
        plotas="contour",
        ncontours=7,
    )

    ax.set_title()

# endregion

#%% theta phase specific extraction of lfp during strong theta MAZE
# region
plt.clf()
fig = plt.figure(1, figsize=(10, 15))
gs = gridspec.GridSpec(2, 3, figure=fig)
fig.subplots_adjust(hspace=0.3)

all_theta = []
cmap = mpl.cm.get_cmap("Set2")
for sub, sess in enumerate(sessions):

    sess.trange = np.array([])
    eegSrate = sess.recinfo.lfpSrate
    maze = sess.epochs.maze

    lfpmaze = sess.utils.geteeg(sess.theta.bestchan, timeRange=maze)
    lfpmaze_t = np.linspace(maze[0], maze[1], len(lfpmaze))

    thetalfp = signal_process.filter_sig.filter_cust(lfpmaze, lf=4, hf=10)
    hil_theta = signal_process.hilbertfast(thetalfp)
    theta_amp = np.abs(hil_theta)

    zsc_theta = stats.zscore(theta_amp)
    thetaevents = threshPeriods(
        zsc_theta, lowthresh=0, highthresh=0.5, minDistance=300, minDuration=1250
    )

    strong_theta = []
    theta_indices = []
    for (beg, end) in thetaevents:
        strong_theta.extend(lfpmaze[beg:end])
        theta_indices.extend(np.arange(beg, end))
    strong_theta = np.asarray(strong_theta)
    theta_indices = np.asarray(theta_indices)
    non_theta = np.delete(lfpmaze, theta_indices)

    theta_lfp = stats.zscore(signal_process.filter_sig.filter_theta(strong_theta))
    # filt_theta = signal_process.filter_sig.filter_cust(theta_lfp, lf=20, hf=60)
    hil_theta = signal_process.hilbertfast(theta_lfp)
    theta_amp = np.abs(hil_theta)
    theta_angle = np.angle(hil_theta, deg=True) + 180
    angle_bin = np.linspace(0, 360, 10)  # divide into 5 bins so each bin=25ms
    bin_ind = np.digitize(theta_angle, bins=angle_bin)

    ax = fig.add_subplot(gs[sub])
    axins = ax.inset_axes([0.5, 0.5, 0.47, 0.47])
    for phase in range(1, len(angle_bin)):
        strong_theta_atphase = theta_lfp[np.where(bin_ind == phase)[0]]
        strong_theta_atphase = signal_process.filter_sig.filter_cust(
            strong_theta_atphase, lf=20, hf=100
        )

        # ax = fig.add_subplot(gs[phase - 1])
        f, t, sxx = sg.spectrogram(
            strong_theta_atphase, nperseg=1250, noverlap=625, fs=1250
        )
        # ax.pcolorfast(t, f, stats.zscore(sxx, axis=1), cmap="YlGn")

        ax.plot(
            f,
            np.mean(sxx, axis=1),
            color=cmap(phase),
            label=f"{int(angle_bin[phase-1])}-{int(angle_bin[phase])}",
        )
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Mean amplitude across time")
        # plt.pcolormesh(bispec_freq, bispec_freq, bispec, vmin=0, vmax=0.1, cmap="YlGn")
        ax.set_xlim([2, 100])

        axins.plot(
            [angle_bin[phase - 1], angle_bin[phase]], [1, 1], color=cmap(phase), lw=2
        )

    axins.axis("off")
    # ax.legend(title="Theta Phase")
    ax.set_title("Mean power spectrum by breaking \n down theta signal by phase")


# fig.suptitle("fourier and bicoherence analysis of strong theta during MAZE")


# endregion

#%% theta phase specific extraction of lfp during REM sleep
# region
plt.clf()
fig = plt.figure(1, figsize=(10, 15))
gs = gridspec.GridSpec(2, 3, figure=fig)
fig.subplots_adjust(hspace=0.3)


cmap = mpl.cm.get_cmap("Set2")
for sub, sess in enumerate(sessions):

    sess.trange = np.array([])
    eegSrate = sess.recinfo.lfpSrate
    maze = sess.epochs.maze

    lfp, _, _ = sess.ripple.best_chan_lfp()
    lfp = lfp[0, :]
    t = np.linspace(0, len(lfp) / 1250, len(lfp))

    tstart = maze[0]
    tend = maze[1]

    lfpmaze = lfp[(t > tstart) & (t < tend)]
    tmaze = np.linspace(tstart, tend, len(lfpmaze))

    frtheta = np.arange(5, 12, 0.5)
    wavdec = signal_process.wavelet_decomp(lfpmaze, freqs=frtheta)
    wav = wavdec.cohen()
    # frgamma = np.arange(25, 50, 1)
    # wavdec = wavelet_decomp(lfpmaze, freqs=frgamma)
    # wav = wavdec.colgin2009()
    # wavtheta = doWavelet(lfpmaze, freqs=frtheta, ncycles=3)

    sum_theta = gaussian_filter1d(np.sum(wav, axis=0), sigma=10)
    zsc_theta = stats.zscore(sum_theta)
    thetaevents = threshPeriods(
        zsc_theta, lowthresh=0, highthresh=0.5, minDistance=300, minDuration=1250
    )

    strong_theta = []
    theta_indices = []
    for (beg, end) in thetaevents:
        strong_theta.extend(lfpmaze[beg:end])
        theta_indices.extend(np.arange(beg, end))
    strong_theta = np.asarray(strong_theta)
    theta_indices = np.asarray(theta_indices)
    non_theta = np.delete(lfpmaze, theta_indices)

    theta_lfp = stats.zscore(signal_process.filter_sig.filter_theta(strong_theta))
    filt_theta = signal_process.filter_sig.filter_cust(theta_lfp, lf=20, hf=60)
    hil_theta = signal_process.hilbertfast(theta_lfp)
    theta_amp = np.abs(hil_theta)
    theta_angle = np.angle(hil_theta, deg=True) + 180
    angle_bin = np.linspace(0, 360, 6)  # divide into 5 bins so each bin=25ms
    bin_ind = np.digitize(theta_angle, bins=angle_bin)

    ax = fig.add_subplot(gs[sub])
    axins = ax.inset_axes([0.5, 0.5, 0.47, 0.47])
    for phase in range(1, len(angle_bin)):
        strong_theta_atphase = strong_theta[np.where(bin_ind == phase)[0]]

        # ax = fig.add_subplot(gs[phase - 1])
        f, t, sxx = sg.spectrogram(
            strong_theta_atphase, nperseg=1250, noverlap=625, fs=1250
        )
        # ax.pcolorfast(t, f, stats.zscore(sxx, axis=1), cmap="YlGn")
        ax.plot(
            f,
            np.mean(sxx, axis=1),
            color=cmap(phase),
            label=f"{int(angle_bin[phase-1])}-{int(angle_bin[phase])}",
        )
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Mean amplitude across time")
        # plt.pcolormesh(bispec_freq, bispec_freq, bispec, vmin=0, vmax=0.1, cmap="YlGn")
        ax.set_xlim([2, 100])

        axins.plot(
            [angle_bin[phase - 1], angle_bin[phase]], [1, 1], color=cmap(phase), lw=2
        )

    axins.axis("off")
    # ax.legend(title="Theta Phase")
    ax.set_title("Mean power spectrum by breaking \n down theta signal by phase")
# endregion

#%% Theta periods and velocity power spectrum with channels at different depths
# region
plt.clf()
fig = plt.figure(1, figsize=(10, 15))
gs = gridspec.GridSpec(2, 3, figure=fig)
fig.subplots_adjust(hspace=0.3)
colors = ["red", "purple", "blue", "green", "k", "orange"]
for sub, sess in enumerate(sessions):

    sess.trange = np.array([])
    eegSrate = sess.recinfo.lfpSrate
    nShanks = sess.recinfo.nShanks
    changrp = sess.recinfo.channelgroups[7]

    posx = sess.position.x
    posy = sess.position.y
    post = sess.position.t
    maze = sess.epochs.maze

    lfp = sess.utils.geteeg(channels=changrp[::3])
    # lfp, _, _ = sess.ripple.best_chan_lfp()
    # lfp = lfp[0, :]
    t = np.linspace(0, lfp.shape[1] / eegSrate, lfp.shape[1])

    tstart = maze[0]
    tend = maze[1]

    # lfpmaze = lfp[(t > tstart) & (t < tend)]
    # tmaze = np.linspace(tstart, tend, len(lfpmaze))
    posmazex = posx[(post > tstart) & (post < tend)]
    posmazey = posy[(post > tstart) & (post < tend)]
    postmaze = np.linspace(tstart, tend, len(posmazex))
    speed = np.sqrt(np.diff(posmazex) ** 2 + np.diff(posmazey) ** 2) / np.diff(postmaze)
    speed = gaussian_filter1d(speed, sigma=10)

    mean_speed = stats.binned_statistic(
        postmaze[:-1], speed, statistic="mean", bins=np.arange(tstart, tend, 1)
    )

    nQuantiles = 8
    quantiles = pd.qcut(mean_speed.statistic, nQuantiles, labels=False)

    alpha_val = np.linspace(0.3, 1, nQuantiles)
    for quantile in range(nQuantiles):
        indx = np.where(quantiles == quantile)[0]
        timepoints = mean_speed.bin_edges[indx]
        lfp_ind = np.concatenate(
            [
                np.arange(int(tstart * 1250), int((tstart + 1) * 1250))
                for tstart in timepoints
            ]
        )
        lfp_quantile = lfp[:, lfp_ind]
        f, pxx = sg.welch(
            lfp_quantile, fs=1250, nperseg=2 * 1250, noverlap=1250, axis=1
        )

        for i in range(6):
            ax = fig.add_subplot(gs[i])
            ax.plot(f, np.log10(pxx[i, :]), color=colors[i], alpha=alpha_val[quantile])
            ax.set_xscale("log")
            # ax.set_yscale("log")
            ax.set_xlim([1, 150])
            ax.set_ylim([2, 6])
# endregion

#%% Selecting channel which shows most theta gamma power at high speed
# region
plt.clf()
fig = plt.figure(1, figsize=(10, 15))
gs = gridspec.GridSpec(2, 3, figure=fig)
fig.subplots_adjust(hspace=0.3)

for sub, sess in enumerate(sessions):

    sess.trange = np.array([])
    eegSrate = sess.recinfo.lfpSrate
    nShanks = sess.recinfo.nShanks
    changrp = sess.recinfo.channelgroups[:nShanks]
    badchans = sess.recinfo.badchans
    goodchans = np.setdiff1d(np.concatenate(changrp), badchans, assume_unique=True)

    posx = sess.position.x
    posy = sess.position.y
    post = sess.position.t
    maze = sess.epochs.maze

    tstart = maze[0]
    tend = maze[1]

    # lfpmaze = lfp[(t > tstart) & (t < tend)]
    # tmaze = np.linspace(tstart, tend, len(lfpmaze))
    posmazex = posx[(post > tstart) & (post < tend)]
    posmazey = posy[(post > tstart) & (post < tend)]
    postmaze = np.linspace(tstart, tend, len(posmazex))
    speed = np.sqrt(np.diff(posmazex) ** 2 + np.diff(posmazey) ** 2) / np.diff(postmaze)
    speed = gaussian_filter1d(speed, sigma=10)

    mean_speed = stats.binned_statistic(
        postmaze[:-1], speed, statistic="mean", bins=np.arange(tstart, tend, 1)
    )

    nQuantiles = 4
    quantiles = pd.qcut(mean_speed.statistic, nQuantiles, labels=False)

    indx = np.where(quantiles == 3)[0]
    timepoints = mean_speed.bin_edges[indx]
    lfp_ind = np.concatenate(
        [
            np.arange(int(tstart * 1250), int((tstart + 1) * 1250))
            for tstart in timepoints
        ]
    )

    auc = []
    for chan in goodchans:
        lfp = sess.utils.geteeg(channels=chan)
        lfp_quantile = lfp[lfp_ind]
        f, pxx = sg.welch(lfp_quantile, fs=1250, nperseg=2 * 1250, noverlap=1250)
        f_theta = np.where((f > 20) & (f < 100))[0]
        area_chan = np.trapz(y=pxx[f_theta], x=f[f_theta])
        auc.append(area_chan)

    # for i in range(len(goodchans)):
    #     ax.plot(f, pxx[i, :])


# endregion

#%% Bicoherence for selected channel at varying speeds
# region
plt.clf()
fig = plt.figure(1, figsize=(10, 15))
gs = gridspec.GridSpec(2, 4, figure=fig)
fig.subplots_adjust(hspace=0.3)

for sub, sess in enumerate(sessions):

    sess.trange = np.array([])
    eegSrate = sess.recinfo.lfpSrate
    nShanks = sess.recinfo.nShanks
    changrp = sess.recinfo.channelgroups[7]

    posx = sess.position.x
    posy = sess.position.y
    post = sess.position.t
    maze = sess.epochs.maze

    lfp = np.asarray(sess.utils.geteeg(chans=125))
    # lfp, _, _ = sess.ripple.best_chan_lfp()
    # lfp = lfp[0, :]
    t = np.linspace(0, len(lfp) / eegSrate, len(lfp))

    tstart = maze[0]
    tend = maze[1]

    # lfpmaze = lfp[(t > tstart) & (t < tend)]
    # tmaze = np.linspace(tstart, tend, len(lfpmaze))
    posmazex = posx[(post > tstart) & (post < tend)]
    posmazey = posy[(post > tstart) & (post < tend)]
    postmaze = np.linspace(tstart, tend, len(posmazex))
    speed = np.sqrt(np.diff(posmazex) ** 2 + np.diff(posmazey) ** 2) / np.diff(postmaze)
    speed = gaussian_filter1d(speed, sigma=10)

    mean_speed = stats.binned_statistic(
        postmaze[:-1], speed, statistic="mean", bins=np.arange(tstart, tend, 1)
    )

    nQuantiles = 8
    quantiles = pd.qcut(mean_speed.statistic, nQuantiles, labels=False)

    for quant, quantile in enumerate([1, 7]):
        indx = np.where(quantiles == quantile)[0]
        timepoints = mean_speed.bin_edges[indx]
        lfp_ind = np.concatenate(
            [
                np.arange(int(tstart * 1250), int((tstart + 1) * 1250))
                for tstart in timepoints
            ]
        )
        lfp_quantile = lfp[lfp_ind]
        bicoh, freq, bispec = signal_process.bicoherence(lfp_quantile, fhigh=90)
        # f, pxx = sg.welch(lfp_quantile, fs=1250, nperseg=2 * 1250, noverlap=1250)

        bicoh = gaussian_filter(np.sqrt(bicoh), sigma=2)
        bicoh = np.where(bicoh > 0.05, bicoh, 0)
        bispec_real = gaussian_filter(np.real(bispec), sigma=2)
        bispec_imag = gaussian_filter(np.imag(bispec), sigma=2)
        bispec_angle = gaussian_filter(np.angle(bispec, deg=True), sigma=2)

        ax = fig.add_subplot(gs[quant, 0])
        im = ax.pcolorfast(freq, freq, bicoh, cmap="Spectral_r", vmax=0.3, vmin=-0.3)
        ax.contour(freq, freq, bicoh, levels=[0.1, 0.2, 0.3], colors="k", linewidths=1)
        ax.set_ylim([1, max(freq) / 2])

        ax = fig.add_subplot(gs[quant, 1])
        ax.pcolorfast(freq, freq, bispec_real, cmap="Spectral_r", vmax=0.3, vmin=-0.3)
        ax.contour(
            freq, freq, bispec_real, levels=[0.1, 0.2, 0.3], colors="k", linewidths=1
        )
        ax.set_ylim([1, max(freq) / 2])

        ax = fig.add_subplot(gs[quant, 2])
        ax.pcolorfast(freq, freq, bispec_imag, cmap="Spectral_r", vmax=0.3, vmin=-0.3)
        ax.contour(
            freq,
            freq,
            bispec_imag,
            levels=[-0.3, -0.2, -0.1],
            colors="k",
            linewidths=1,
        )
        ax.set_ylim([1, max(freq) / 2])

        ax = fig.add_subplot(gs[quant, 3])
        ang = ax.pcolorfast(freq, freq, bispec_angle, cmap="bwr", vmax=180, vmin=-180)
        ax.set_ylim([1, max(freq) / 2])
    cax = plt.axes([0.05, 0.88, 0.3, 0.2])
    fig.colorbar(im, ax=cax, orientation="horizontal")
    cax.axis("off")

    cax = plt.axes([0.65, 0.88, 0.3, 0.2])
    fig.colorbar(ang, ax=cax, orientation="horizontal")
    cax.axis("off")

# endregion

#%% Power-Power correlation
# region


plt.clf()
fig = plt.figure(1, figsize=(10, 15))
gs = gridspec.GridSpec(2, 3, figure=fig)
fig.subplots_adjust(hspace=0.3)
colors = ["red", "purple", "blue", "green", "k", "orange"]
for sub, sess in enumerate(sessions):

    sess.trange = np.array([])
    eegSrate = sess.recinfo.lfpSrate
    nShanks = sess.recinfo.nShanks
    changrp = sess.recinfo.channelgroups[7]

    posx = sess.position.x
    posy = sess.position.y
    post = sess.position.t
    maze = sess.epochs.maze

    lfp = sess.utils.geteeg(channels=changrp[::3])
    # lfp, _, _ = sess.ripple.best_chan_lfp()
    # lfp = lfp[0, :]
    t = np.linspace(0, lfp.shape[1] / eegSrate, lfp.shape[1])

    tstart = maze[0]
    tend = maze[1]

    # lfpmaze = lfp[(t > tstart) & (t < tend)]
    # tmaze = np.linspace(tstart, tend, len(lfpmaze))
    posmazex = posx[(post > tstart) & (post < tend)]
    posmazey = posy[(post > tstart) & (post < tend)]
    postmaze = np.linspace(tstart, tend, len(posmazex))
    speed = np.sqrt(np.diff(posmazex) ** 2 + np.diff(posmazey) ** 2) / np.diff(postmaze)
    speed = gaussian_filter1d(speed, sigma=10)

    mean_speed = stats.binned_statistic(
        postmaze[:-1], speed, statistic="mean", bins=np.arange(tstart, tend, 1)
    )

    nQuantiles = 8
    quantiles = pd.qcut(mean_speed.statistic, nQuantiles, labels=False)

    alpha_val = np.linspace(0.3, 1, nQuantiles)
    for quantile in [7]:
        indx = np.where(quantiles == quantile)[0]
        timepoints = mean_speed.bin_edges[indx]
        lfp_ind = np.concatenate(
            [
                np.arange(int(tstart * 1250), int((tstart + 1) * 1250))
                for tstart in timepoints
            ]
        )
        lfp_quantile = lfp[:, lfp_ind]

        for i in range(6):
            f, t, sxx = sg.spectrogram(
                lfp_quantile[i, :], fs=1250, nperseg=2 * 1250, noverlap=1250
            )
            corr_quantile = np.corrcoef(sxx)
            f_req = np.where(f < 120)[0]
            ax = fig.add_subplot(gs[i])
            ax.pcolorfast(
                f[f_req], f[f_req], corr_quantile[np.ix_(f_req, f_req)], vmax=0.3
            )
            # ax.set_xscale("log")
            # # ax.set_yscale("log")
            # ax.set_xlim([1, 150])
            # ax.set_ylim([2, 6])
# endregion

#%% Slow Gamma/Spectrogram for REM sleep theta oscillation
# region
plt.clf()
fig = plt.figure(1, figsize=(10, 15))
gs = gridspec.GridSpec(1, 6, figure=fig)
fig.subplots_adjust(hspace=0.3)
for sub, sess in enumerate(sessions):

    sess.trange = np.array([])
    eegSrate = sess.recinfo.lfpSrate
    states = sess.brainstates.states
    rems = states.loc[states["name"] == "rem"]

    lfp = sess.theta.getBestChanlfp()
    if sub in [1, 4]:
        lfp = sess.utils.geteeg(chans=50)

    rem_frames = []
    for rem in rems.itertuples():
        rem_frames.extend(
            list(range(int(rem.start * eegSrate), int(rem.end * eegSrate)))
        )
    rem_theta = lfp[rem_frames]

    # -----wavelet computation -------
    frgamma = np.arange(25, 150, 1)
    wavdec = signal_process.wavelet_decomp(rem_theta, freqs=frgamma)
    wav = wavdec.colgin2009()
    # wav = wavdec.cohen(ncycles=7)
    wav = stats.zscore(wav, axis=1)

    # ---phase calculation -----------
    theta_filter = stats.zscore(
        signal_process.filter_sig.filter_cust(rem_theta, lf=4, hf=11)
    )
    hil_theta = signal_process.hilbertfast(theta_filter)
    theta_amp = np.abs(hil_theta)
    theta_angle = np.angle(hil_theta, deg=True) + 180
    theta_troughs = sg.find_peaks(-theta_filter)[0]
    bin_angle = np.linspace(0, 360, int(360 / 9) + 1)
    bin_ind = np.digitize(theta_angle, bin_angle)

    wav_phase = []
    for i in np.unique(bin_ind):
        find_where = np.where(bin_ind == i)[0]
        wav_at_angle = np.mean(wav[:, find_where], axis=1)
        wav_phase.append(wav_at_angle)

    wav_phase = np.asarray(wav_phase).T

    # wav_theta_all = np.dstack(wav_theta_all).mean(axis=2)

    ax = fig.add_subplot(gs[sub])
    ax.clear()
    im = ax.pcolorfast(bin_angle[:-1], frgamma[:-1], wav_phase, cmap="Spectral_r")
    ax.set_xlabel(r"$\theta$ phase")
    ax.set_ylabel("frequency (Hz)")
    fig.colorbar(im, ax=ax, orientation="horizontal")

# endregion

#%% Theta-Gamma band correlation during SD
# region
plt.clf()
fig = plt.figure(1, figsize=(10, 15))
gs = gridspec.GridSpec(3, 1, figure=fig)
fig.subplots_adjust(hspace=0.3)

for sub, sess in enumerate(sessions[:3]):
    sess.trange = np.array([])
    specpower = sess.brainstates.params

    sd_period = sess.epochs.post
    specpower_sd = specpower.loc[
        (specpower.time > sd_period[0]) & (specpower.time < sd_period[0] + 5 * 3600)
    ]
    specpower_sd1st = specpower.loc[
        (specpower.time > sd_period[0]) & (specpower.time < sd_period[0] + 3600)
    ]
    specpower_sd5th = specpower.loc[
        (specpower.time > sd_period[0] + 4 * 3600)
        & (specpower.time < sd_period[0] + 5 * 3600)
    ]

    corr_1st = np.corrcoef(
        specpower_sd1st.theta_deltaplus_ratio, specpower_sd1st.gamma
    )[0, 1]
    corr_5th = np.corrcoef(
        specpower_sd5th.theta_deltaplus_ratio, specpower_sd5th.gamma
    )[0, 1]

    print(corr_1st, corr_5th)
    # thdel_ratio = np.asarray(specpower_sd.theta_deltaplus_ratio)
    # gamma = np.asarray(specpower_sd.gamma)
    # ax = fig.add_subplot(gs[sub])
    # ax.plot(
    #     gamma / np.nansum(gamma),
    #     thdel_ratio / np.nansum(thdel_ratio),
    #     ".",
    #     markersize=0.5,
    # )


# endregion

#%% Strong theta periods during SD wavlet spectrogram around in gamma band
# region
plt.clf()
fig = plt.figure(1, figsize=(8, 11))
gs = gridspec.GridSpec(3, 1, figure=fig)
fig.subplots_adjust(hspace=0.3)

chans = [13, 25, 39]
for sub, sess in enumerate(sessions[:3]):
    sess.trange = np.array([])
    post = sess.epochs.post
    sd_period = [post[0] + 3 * 3600, post[0] + 5 * 3600]
    eeg = sess.utils.geteeg(chans=chans[sub], timeRange=sd_period)

    specgram = signal_process.spectrogramBands(eeg, window=1, overlap=0.5, smooth=10)
    zsc_theta = stats.zscore(specgram.theta)

    thetaevents = threshPeriods(
        zsc_theta, lowthresh=0, highthresh=1, minDistance=300, minDuration=1250
    )

    strong_theta = []
    theta_indices = []
    for (beg, end) in thetaevents:
        strong_theta.extend(eeg[beg:end])
        theta_indices.extend(np.arange(beg, end))
    strong_theta = np.asarray(strong_theta)
    theta_indices = np.asarray(theta_indices)

    # non_theta = np.delete(lfpSD, theta_indices)
    frgamma = np.arange(25, 150, 1)
    # frgamma = np.linspace(25, 150, 1)
    strong_theta = np.asarray(strong_theta)
    wavdec = signal_process.wavelet_decomp(strong_theta, freqs=frgamma)
    wav = wavdec.colgin2009()
    # wav = wavdec.cohen(ncycles=7)

    wav = stats.zscore(wav, axis=1)

    theta_filter = stats.zscore(
        signal_process.filter_sig.filter_cust(strong_theta, lf=4, hf=11)
    )

    hil_theta = signal_process.hilbertfast(theta_filter)
    theta_amp = np.abs(hil_theta)
    theta_angle = np.angle(hil_theta, deg=True) + 180
    theta_troughs = sg.find_peaks(-theta_filter)[0]

    bin_angle = np.linspace(0, 360, int(360 / 9) + 1)
    bin_ind = np.digitize(theta_angle, bin_angle)

    wav_phase = []
    for i in np.unique(bin_ind):
        find_where = np.where(bin_ind == i)[0]
        wav_at_angle = np.mean(wav[:, find_where], axis=1)
        wav_phase.append(wav_at_angle)

    wav_phase = np.asarray(wav_phase).T

    ax = fig.add_subplot(gs[sub])
    ax.pcolorfast(bin_angle[:-1], frgamma[:-1], wav_phase, cmap="Spectral_r")
    ax.set_xlabel(r"$\theta$ phase")

    ax.set_ylabel("frequency (Hz)")

    # bicoh, freq, bispec = signal_process.bicoherence(strong_theta, fhigh=100)

    # # bicoh = gaussian_filter(bicoh, sigma=2)
    # # bicoh = np.where(bicoh > 0.05, bicoh, 0)
    # bispec_real = gaussian_filter(np.real(bispec), sigma=2)
    # bispec_imag = gaussian_filter(np.imag(bispec), sigma=2)
    # bispec_angle = gaussian_filter(np.angle(bispec, deg=True), sigma=2)

    # ax = fig.add_subplot(gs[0, 1])
    # ax.clear()
    # im = ax.pcolorfast(freq, freq, bicoh, cmap="Spectral_r", vmax=0.05, vmin=0)
    # # ax.contour(freq, freq, bicoh, levels=[0.1, 0.2, 0.3], colors="k", linewidths=1)
    # ax.set_ylim([1, max(freq) / 2])
    # ax.set_xlabel("Frequency (Hz)")
    # ax.set_ylabel("Frequency (Hz)")

    # # cax = fig.add_axes([0.3, 0.8, 0.5, 0.05])
    # # cax.clear()
    # # ax.contour(freq, freq, bicoh, levels=[0.1, 0.2, 0.3], colors="k", linewidths=1)
    # fig.colorbar(im, ax=ax, orientation="horizontal")


# endregion


#%% Multiple regression analysis on slow gamma power explained by variables such as theta-harmonic, theta-asymmetry, speed etc. Also comparing it with theta-harmonic being explained by similar variables
# region
plt.clf()
fig, ax = plt.subplots(1, 2, num=1, sharey=True)
exp_var_gamma_all, exp_var_harmonic_all = [], []
for sub, sess in enumerate(sessions):
    sess.trange = np.array([])
    maze = sess.epochs.maze
    speed = sess.position.speed
    t_position = sess.position.t[1:]
    deadtime = sess.artifact.time

    lfpmaze = sess.utils.geteeg(sess.theta.bestchan, timeRange=maze)
    lfpmaze_t = np.linspace(maze[0], maze[1], len(lfpmaze))
    speed = np.interp(lfpmaze_t, t_position, speed)

    if deadtime is not None:
        dead_indx = np.concatenate(
            [
                np.where((lfpmaze_t > start) & (lfpmaze_t < end))[0]
                for (start, end) in deadtime
            ]
        )
        lfpmaze = np.delete(lfpmaze, dead_indx)
        speed = np.delete(speed, dead_indx)

    # --- calculating theta parameters ---------
    thetalfp = signal_process.filter_sig.filter_cust(lfpmaze, lf=4, hf=10)
    hil_theta = signal_process.hilbertfast(thetalfp)
    theta_angle = np.abs(np.angle(hil_theta, deg=True))
    theta_trough = sg.find_peaks(theta_angle)[0]
    theta_peak = sg.find_peaks(-theta_angle)[0]
    theta_amp = np.abs(hil_theta) ** 2

    # --- calculating slow gamma parameters -------
    gammalfp = signal_process.filter_sig.filter_cust(lfpmaze, lf=25, hf=50)
    hil_gamma = signal_process.hilbertfast(gammalfp)
    gamma_amp = np.abs(hil_gamma) ** 2

    # --- theta harmonic ----------
    theta_harmonic = signal_process.filter_sig.filter_cust(lfpmaze, lf=10, hf=22)
    hil_theta_harmonic = signal_process.hilbertfast(theta_harmonic)
    theta_harmonic_amp = np.abs(hil_theta_harmonic) ** 2

    if theta_peak[0] < theta_trough[0]:
        theta_peak = theta_peak[1:]
    if theta_trough[-1] > theta_peak[-1]:
        theta_trough = theta_trough[:-1]

    assert len(theta_trough) == len(theta_peak)

    rising_time = (theta_peak[1:] - theta_trough[1:]) / 1250
    falling_time = (theta_trough[1:] - theta_peak[:-1]) / 1250
    rise_fall = rising_time / (rising_time + falling_time)

    rise_midpoints = np.array(
        [
            trough
            + np.argmin(
                np.abs(
                    thetalfp[trough:peak]
                    - (max(thetalfp[trough:peak]) - np.ptp(thetalfp[trough:peak]) / 2)
                )
            )
            for (trough, peak) in zip(theta_trough, theta_peak)
        ]
    )

    fall_midpoints = np.array(
        [
            peak
            + np.argmin(
                np.abs(
                    thetalfp[peak:trough]
                    - (max(thetalfp[peak:trough]) - np.ptp(thetalfp[peak:trough]) / 2)
                )
            )
            for (peak, trough) in zip(theta_peak[:-1], theta_trough[1:])
        ]
    )
    peak_width = fall_midpoints - rise_midpoints[:-1]
    trough_width = rise_midpoints[1:] - fall_midpoints
    peak_trough_asymm = peak_width / (peak_width + trough_width)

    speed_in_theta = stats.binned_statistic(
        np.arange(len(thetalfp)), speed, bins=theta_trough
    )[0]
    thetapower_in_theta = stats.binned_statistic(
        np.arange(len(thetalfp)), theta_amp, bins=theta_trough
    )[0]
    gammapower_in_theta = stats.binned_statistic(
        np.arange(len(thetalfp)), gamma_amp, bins=theta_trough
    )[0]
    thetaharmonicpower_in_theta = stats.binned_statistic(
        np.arange(len(thetalfp)), theta_harmonic_amp, bins=theta_trough
    )[0]

    data = pd.DataFrame(
        {
            "gammaPower": gammapower_in_theta,
            "thetaharmonicPower": thetaharmonicpower_in_theta,
            "thetaPower": thetapower_in_theta,
            "speed": speed_in_theta,
            "asymm": rise_fall,
            "peaktrough": peak_trough_asymm,
        }
    )

    variables = data.columns.tolist()[2:]
    par_corr_stats_gamma = [
        data.partial_corr(
            y="gammaPower", x=var, covar=list(set(variables) - set([var]))
        )
        for var in variables
    ]
    par_corr_stats_harmonic = [
        data.partial_corr(
            y="thetaharmonicPower", x=var, covar=list(set(variables) - set([var]))
        )
        for var in variables
    ]

    exp_var_gamma = np.array([stat_.r2[0] * 100 for stat_ in par_corr_stats_gamma])
    p_val_gamma = np.array([stat_["p-val"][0] for stat_ in par_corr_stats_gamma])

    exp_var_harmonic = np.array(
        [stat_.r2[0] * 100 for stat_ in par_corr_stats_harmonic]
    )
    p_val_harmonic = np.array([stat_["p-val"][0] for stat_ in par_corr_stats_harmonic])

    exp_var_gamma_all.append(
        pd.DataFrame(exp_var_gamma[np.newaxis, :], columns=variables)
    )
    exp_var_harmonic_all.append(
        pd.DataFrame(exp_var_harmonic[np.newaxis, :], columns=variables)
    )


exp_var_gamma_all = pd.concat(exp_var_gamma_all)
# sns.barplot(ax=ax[0], data=exp_var_gamma_all, ci=None)
ax[0].bar(
    exp_var_gamma_all.columns.tolist(),
    exp_var_gamma_all.mean().values,
    # fmt="None",
    yerr=exp_var_gamma_all.sem().values,
    ecolor="black",
    capsize=10,
    edgecolor="k",
)
ax[0].tick_params(axis="x", labelrotation=90)
ax[0].set_ylabel("Explained variance (%)")
ax[0].set_title("Slow-gamma")

exp_var_harmonic_all = pd.concat(exp_var_harmonic_all)
# sns.barplot(ax=ax[1], data=exp_var_harmonic_all, ci=None)
ax[1].bar(
    exp_var_harmonic_all.columns.tolist(),
    exp_var_harmonic_all.mean().values,
    yerr=exp_var_harmonic_all.sem().values,
    ecolor="black",
    capsize=10,
    edgecolor="k",
)
ax[1].tick_params(axis="x", labelrotation=90)
ax[1].set_title("Theta harmonic (10-22 Hz)")


# endregion

