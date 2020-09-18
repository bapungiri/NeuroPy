# %%
import os
import random
import warnings

import ipywidgets as widgets
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import scipy.signal as sg
import scipy.stats as stats
import seaborn as sns
import statsmodels.api as sm
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from sklearn import linear_model

import signal_process
from callfunc import processData
from mathutil import threshPeriods
from plotUtil import Fig

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
    "/data/Clustering/SleepDeprivation/RatJ/Day1/",
    "/data/Clustering/SleepDeprivation/RatK/Day1/",
    "/data/Clustering/SleepDeprivation/RatN/Day1/",
    "/data/Clustering/SleepDeprivation/RatJ/Day2/",
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

#%% theta phase specific extraction of lfp during strong theta MAZE with different binning techiques
# region

figure = Fig()
fig, gs = figure.draw(grid=[4, 3])

axbin1 = plt.subplot(gs[1, 0])
axbin1.clear()
figure.panel_label(axbin1, "b")
axbin2 = plt.subplot(gs[1, 1])
axbin2.clear()
axslide = plt.subplot(gs[1, 2])
axslide.clear()

all_theta = []
bin1Data = pd.DataFrame()
bin2Data = pd.DataFrame()
slideData = pd.DataFrame()
cmap = mpl.cm.get_cmap("Set3")
for sub, sess in enumerate(sessions[3:5]):

    sess.trange = np.array([])
    eegSrate = sess.recinfo.lfpSrate
    maze = sess.epochs.maze

    lfpmaze = sess.utils.geteeg(sess.theta.bestchan, timeRange=maze)
    lfpmaze_t = np.linspace(maze[0], maze[1], len(lfpmaze))

    # ---- filtering --> zscore --> threshold --> strong theta periods ----
    thetalfp = signal_process.filter_sig.bandpass(lfpmaze, lf=4, hf=10)
    hil_theta = signal_process.hilbertfast(thetalfp)
    theta_amp = np.abs(hil_theta)

    zsc_theta = stats.zscore(theta_amp)
    thetaevents = threshPeriods(
        zsc_theta, lowthresh=0, highthresh=0.5, minDistance=300, minDuration=1250
    )

    strong_theta, theta_indices = [], []
    for (beg, end) in thetaevents:
        strong_theta.extend(lfpmaze[beg:end])
        theta_indices.extend(np.arange(beg, end))
    strong_theta = np.asarray(strong_theta)
    theta_indices = np.asarray(theta_indices)
    non_theta = np.delete(lfpmaze, theta_indices)

    # ---- filtering strong theta periods into theta and gamma band ------
    theta_lfp = stats.zscore(
        signal_process.filter_sig.bandpass(strong_theta, lf=4, hf=10)
    )
    gamma_lfp = stats.zscore(
        signal_process.filter_sig.highpass(strong_theta, cutoff=25)
    )

    # ----- phase detection for theta band -----------
    # filt_theta = signal_process.filter_sig.filter_cust(theta_lfp, lf=20, hf=60)
    hil_theta = signal_process.hilbertfast(theta_lfp)
    theta_amp = np.abs(hil_theta)
    theta_angle = np.angle(hil_theta, deg=True) + 180  # range from 0 to 360

    """
    phase specific extraction of highpass filtered strong theta periods (>25 Hz) and concatenating similar phases across multiple theta cycles
    """

    # ----- dividing 360 degress into non-overlapping 5 bins ------------
    angle_bin = np.linspace(0, 360, 6)  # 5 bins so each bin=25ms
    angle_centers = angle_bin + np.diff(angle_bin).mean() / 2
    bin_ind = np.digitize(theta_angle, bins=angle_bin)
    df1 = pd.DataFrame()
    for phase in range(1, len(angle_bin)):
        strong_theta_atphase = gamma_lfp[np.where(bin_ind == phase)[0]]
        f_, pxx = sg.welch(strong_theta_atphase, nperseg=1250, noverlap=625, fs=1250)
        df1["freq"] = f_
        df1[str(angle_centers[phase - 1])] = pxx
    bin1Data = bin1Data.append(df1)

    # ----- dividing 360 degress into non-overlapping 9 bins ------------
    angle_bin = np.linspace(0, 360, 10)  # 9 bins
    angle_centers = angle_bin + np.diff(angle_bin).mean() / 2
    bin_ind = np.digitize(theta_angle, bins=angle_bin)
    df2 = pd.DataFrame()
    for phase in range(1, len(angle_bin)):
        strong_theta_atphase = gamma_lfp[np.where(bin_ind == phase)[0]]
        f_, pxx = sg.welch(strong_theta_atphase, nperseg=1250, noverlap=625, fs=1250)
        df2["freq"] = f_
        df2[str(angle_centers[phase - 1])] = pxx
    bin2Data = bin2Data.append(df2)

    # ----- dividing 360 degress into sliding windows ------------
    window = 40  # degress
    slideby = 5  # degress
    angle_bin = np.arange(0, 360 - 40, slideby)  # divide into 5 bins so each bin=25ms
    angle_centers = angle_bin + window / 2
    bin_ind = np.digitize(theta_angle, bins=angle_bin)
    df3 = pd.DataFrame()
    for phase in angle_bin:
        strong_theta_atphase = gamma_lfp[
            np.where((theta_angle > phase) & (theta_angle < phase + window))[0]
        ]
        f_, pxx = sg.welch(strong_theta_atphase, nperseg=1250, noverlap=625, fs=1250)
        df3["freq"] = f_
        df3[str(phase + window / 2)] = pxx
    slideData = slideData.append(df3)


mean_bin1 = bin1Data.groupby(level=0).mean()
mean_bin2 = bin2Data.groupby(level=0).mean()
mean_slide = slideData.groupby(level=0).mean()

mean_bin1.plot(x="freq", ax=axbin1, legend=False, linewidth=1)
mean_bin2.plot(x="freq", ax=axbin2, legend=False, linewidth=1)
mean_slide.plot(x="freq", ax=axslide, legend=False, linewidth=1)

# ---- figure properties ----------
[
    [ax.set_xlabel("Frequency (Hz)"), ax.set_ylabel("Power"), ax.set_xlim([0, 200])]
    for ax in [axbin1, axbin2, axslide]
]
[
    ax.set_title(title)
    for (ax, title) in zip(
        [axbin1, axbin2, axslide], ["5 bins", "9 bins", "sliding window"]
    )
]
axbin1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
axbin2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
axslide.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

scriptname = os.path.basename(__file__)
filename = "phase_specific_slowgamma"
figure.savefig(filename, scriptname)


# endregion

#%% theta phase specific extraction of lfp during REM sleep
# region

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
plt.clf()
fig = plt.figure(1, figsize=(10, 15))
gs = gridspec.GridSpec(2, 2, figure=fig)
fig.subplots_adjust(hspace=0.3)

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
    thetalfp = signal_process.filter_sig.bandpass(lfpmaze, lf=1, hf=25)
    hil_theta = signal_process.hilbertfast(thetalfp)
    theta_angle = np.abs(np.angle(hil_theta, deg=True))
    theta_trough = sg.find_peaks(theta_angle)[0]
    theta_peak = sg.find_peaks(-theta_angle)[0]
    theta_amp = np.abs(hil_theta) ** 2

    # --- calculating slow gamma parameters -------
    gammalfp = signal_process.filter_sig.bandpass(lfpmaze, lf=25, hf=50)
    hil_gamma = signal_process.hilbertfast(gammalfp)
    gamma_amp = np.abs(hil_gamma) ** 2

    # --- theta harmonic ----------
    theta_harmonic = signal_process.filter_sig.bandpass(lfpmaze, lf=10, hf=22)
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
    peak_width = (fall_midpoints - rise_midpoints[:-1]) / 1250
    trough_width = (rise_midpoints[1:] - fall_midpoints) / 1250
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

    variables1 = data.columns.tolist()[1:]
    par_corr_stats_gamma = [
        data.partial_corr(
            y="gammaPower", x=var, covar=list(set(variables1) - set([var]))
        )
        for var in variables1
    ]

    variables2 = data.columns.tolist()[2:]
    par_corr_stats_harmonic = [
        data.partial_corr(
            y="thetaharmonicPower", x=var, covar=list(set(variables2) - set([var]))
        )
        for var in variables2
    ]

    exp_var_gamma = np.array([stat_.r2[0] * 100 for stat_ in par_corr_stats_gamma])
    p_val_gamma = np.array([stat_["p-val"][0] for stat_ in par_corr_stats_gamma])

    exp_var_harmonic = np.array(
        [stat_.r2[0] * 100 for stat_ in par_corr_stats_harmonic]
    )
    p_val_harmonic = np.array([stat_["p-val"][0] for stat_ in par_corr_stats_harmonic])

    exp_var_gamma_all.append(
        pd.DataFrame(exp_var_gamma[np.newaxis, :], columns=variables1)
    )
    exp_var_harmonic_all.append(
        pd.DataFrame(exp_var_harmonic[np.newaxis, :], columns=variables2)
    )


exp_var_gamma_all = pd.concat(exp_var_gamma_all)
# sns.barplot(ax=ax[0], data=exp_var_gamma_all, ci=None)
ax1 = plt.subplot(gs[1, 0])
ax1.bar(
    exp_var_gamma_all.columns.tolist(),
    exp_var_gamma_all.mean().values,
    # fmt="None",
    yerr=exp_var_gamma_all.sem().values,
    ecolor="black",
    capsize=10,
    edgecolor="k",
    color="#ffa69e",
)
ax1.tick_params(axis="x", labelrotation=90)
ax1.set_ylabel("Explained variance (%)")
ax1.set_title("Slow-gamma")

exp_var_harmonic_all = pd.concat(exp_var_harmonic_all)
# sns.barplot(ax=ax[1], data=exp_var_harmonic_all, ci=None)
ax2 = plt.subplot(gs[1, 1], sharey=ax1)
ax2.bar(
    exp_var_harmonic_all.columns.tolist(),
    exp_var_harmonic_all.mean().values,
    yerr=exp_var_harmonic_all.sem().values,
    ecolor="black",
    capsize=10,
    edgecolor="k",
    color="#ffbf00",
)
ax2.tick_params(axis="x", labelrotation=90)
ax2.set_title("Theta harmonic (10-22 Hz)")


# endregion
