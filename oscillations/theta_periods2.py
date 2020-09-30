# %%
import os
import random
from typing import Dict
import warnings

# import ipywidgets as widgets
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import scipy.signal as sg
import scipy.stats as stats
import seaborn as sns
import signal_process
import statsmodels.api as sm
from callfunc import processData
from mathutil import threshPeriods
from plotUtil import Fig, Colormap
from scipy import fft
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from sklearn import linear_model
from tables.description import Col

# warnings.simplefilter(action="default")

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
    "/data/Clustering/SleepDeprivation/RatK/Day2/",
    "/data/Clustering/SleepDeprivation/RatN/Day2/",
    # "/data/Clustering/SleepDeprivation/RatJ/Day4/",
    "/data/Clustering/SleepDeprivation/RatK/Day4/",
    # "/data/Clustering/SleepDeprivation/RatN/Day4/",
    "/data/Clustering/SleepDeprivation/RatA14d1LP/Rollipram/",
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

#%%* theta phase specific extraction of lfp during strong theta MAZE with different binning techiques
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

for sub, sess in enumerate(sessions[4:5]):

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
        signal_process.filter_sig.bandpass(strong_theta, lf=1, hf=25)
    )
    gamma_lfp = stats.zscore(
        signal_process.filter_sig.highpass(strong_theta, cutoff=25, order=3)
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

# figure.savefig("phase_specific_slowgamma", __file__)


# endregion

#%% theta phase specific extraction of lfp during HIGH VELOCITY epochs on MAZE theta MAZE with different binning techiques
# region

data: Dict[str, np.array] = {}

for sub, sess in enumerate(sessions[6:7]):

    sess.trange = np.array([])
    eegSrate = sess.recinfo.lfpSrate
    changrp = sess.recinfo.goodchangrp
    maze = sess.epochs.maze
    speed = sess.position.speed
    t_position = sess.position.t[1:]
    chans2plot = np.concatenate([shank[::6] for shank in changrp]).astype(int)
    shank = [
        shank
        for shank in range(len(changrp))
        for chan in chans2plot
        if chan in changrp[shank]
    ]

    lfpmaze = sess.utils.geteeg(chans=chans2plot, timeRange=maze)
    lfpmaze_t = np.linspace(maze[0], maze[1], lfpmaze.shape[-1])
    speed = np.interp(lfpmaze_t, t_position, speed)
    speed = gaussian_filter1d(speed, sigma=10)

    frames_high_spd = np.where(speed > 25)[0]
    lfp_highspd = lfpmaze[:, frames_high_spd]

    frames_slow_spd = np.where(speed <= 25)[0]
    lfp_lowspd = lfpmaze[:, frames_slow_spd]

    # ---- filtering strong theta periods into theta and gamma band ------
    # theta_lfp = stats.zscore(
    #     signal_process.filter_sig.bandpass(lfp_highspd, lf=1, hf=25, ax=-1)
    # )

    # gamma_lfp = stats.zscore(
    #     signal_process.filter_sig.highpass(lfp_highspd, cutoff=25, order=3, ax=-1)
    # )

    # ----- phase detection for theta band -----------
    # filt_theta = signal_process.filter_sig.filter_cust(theta_lfp, lf=20, hf=60)
    # hil_theta = signal_process.hilbertfast(theta_lfp)
    # theta_amp = np.abs(hil_theta)
    # theta_angle = np.angle(hil_theta, deg=True) + 180  # range from 0 to 360

    # ------ psd calculation-----------
    f_, pxx = sg.welch(lfp_highspd, fs=1250, nperseg=4 * 1250, noverlap=2 * 250)
    f_slow, pxx_slow = sg.welch(lfp_lowspd, fs=1250, nperseg=4 * 1250, noverlap=2 * 250)

    # ---- bicoherence calculation ----------
    bicoh, f, _ = signal_process.bicoherence_m(lfp_highspd, flow=1, fhigh=70)

    data[sub] = {
        "chans": chans2plot,
        "fpxx_slow": f_slow,
        "pxx_slow": pxx_slow,
        "fpxx": f_,
        "pxx": pxx,
        "fbicoh": f,
        "bicoh": bicoh,
    }


# ---- plotting ----------
figure = Fig()
cmap = Colormap().dynamic3()
for i in range(len(data)):
    data_sub = data[i]
    fig, gs = figure.draw(num=i + 1, grid=[7, 8], size=[15, 15])
    for chan in range(len(data_sub["chans"])):
        ax = plt.subplot(gs[2 * chan])
        ax.plot(data_sub["fpxx"], data_sub["pxx"][chan])
        ax.plot(data_sub["fpxx_slow"], data_sub["pxx_slow"][chan])
        ax.set_ylabel("Power")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim([3, 200])
        ax.set_ylim(bottom=10)

        ax = plt.subplot(gs[2 * chan + 1])
        # ax.imshow(data_sub["bicoh"][chan, :, :])
        bic = data_sub["bicoh"][chan, :, :]
        bic = np.sqrt(bic)
        lt = np.tril_indices_from(bic, k=-1)
        bic[lt] = np.nan
        bic[(lt[0], -lt[1])] = np.nan
        bic = bic - np.nanmean(bic)
        bic[bic < 0.1] = 0
        # bic = stats.mstats.zscore(bic, nan_policy="omit")
        # bic = gaussian_filter(bic, sigma=0.5)
        bicoh_plt = ax.pcolormesh(
            data_sub["fbicoh"],
            data_sub["fbicoh"],
            bic,
            cmap=cmap,
            # shading="gouraud",
            vmin=-0.2,
            vmax=0.2,
        )

        ax.set_ylim([0, np.max(data_sub["fbicoh"]) / 2])

        ax.plot(
            [1, np.max(data_sub["fbicoh"]) / 2],
            [1, np.max(data_sub["fbicoh"]) / 2],
            "gray",
        )
        ax.plot(
            [np.max(data_sub["fbicoh"]) / 2, np.max(data_sub["fbicoh"])],
            [np.max(data_sub["fbicoh"]) / 2, 1],
            "gray",
        )
        # ax.set_title(sessions[i].sessinfo.session.sessionName)


# figure.savefig("phase_specific_slowgamma", __file__)


# endregion

#%% bicoherence at multiple depths of linear probe
# region

data: Dict[str, np.array] = {}

for sub, sess in enumerate(sessions[7:8]):

    sess.trange = np.array([])
    eegSrate = sess.recinfo.lfpSrate
    changrp = sess.recinfo.goodchangrp
    # chans2plot = np.concatenate([shank[::14] for shank in changrp]).astype(int)
    chans2plot = np.array([94, 112, 126]).astype(int)
    shank = [
        shank
        for shank in range(len(changrp))
        for chan in chans2plot
        if chan in changrp[shank]
    ]
    maze = sess.epochs.maze

    lfpmaze = sess.utils.geteeg(chans=63, timeRange=maze)
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

    lfpmaze = sess.utils.geteeg(chans=chans2plot, timeRange=maze)
    lfpmaze_t = np.linspace(maze[0], maze[1], lfpmaze.shape[-1])

    lfp_highspd = lfpmaze[:, theta_indices]

    # ------ psd calculation-----------
    f_, pxx = sg.welch(lfp_highspd, fs=1250, nperseg=4 * 1250, noverlap=2 * 250)
    # f_slow, pxx_slow = sg.welch(lfp_lowspd, fs=1250, nperseg=4 * 1250, noverlap=2 * 250)

    # ---- bicoherence calculation ----------
    bicoh, f, _ = signal_process.bicoherence_m(lfp_highspd, flow=1, fhigh=180)

    data[sub] = {
        "chans": chans2plot,
        # "fpxx_slow": f_slow,
        # "pxx_slow": pxx_slow,
        "fpxx": f_,
        "pxx": pxx,
        "fbicoh": f,
        "bicoh": bicoh,
    }


# ---- plotting ----------
figure = Fig()
cmap = Colormap().dynamic3()
for i in range(len(data)):
    data_sub = data[i]
    fig, gs = figure.draw(num=i + 1, grid=[3, 2], size=[15, 15])
    for chan in range(len(data_sub["chans"])):
        ax = plt.subplot(gs[2 * chan])
        ax.plot(data_sub["fpxx"], data_sub["pxx"][chan])
        # ax.plot(data_sub["fpxx_slow"], data_sub["pxx_slow"][chan])
        ax.set_ylabel("Power")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim([3, 200])
        ax.set_ylim(bottom=10)

        ax = plt.subplot(gs[2 * chan + 1])
        # ax.imshow(data_sub["bicoh"][chan, :, :])
        bic = data_sub["bicoh"][chan, :, :]
        bic = np.sqrt(bic)
        lt = np.tril_indices_from(bic, k=-1)
        bic[lt] = np.nan
        bic[(lt[0], -lt[1])] = np.nan
        bic = bic - np.nanmean(bic)
        # bic[bic < 0.1] = 0
        # bic = stats.mstats.zscore(bic, nan_policy="omit")
        bic = gaussian_filter(bic, sigma=1)
        bicoh_plt = ax.pcolormesh(
            data_sub["fbicoh"],
            data_sub["fbicoh"],
            bic,
            cmap="Spectral_r",
            # shading="gouraud",
            vmin=-0.1,
            vmax=0.1,
        )

        ax.set_ylim([0, np.max(data_sub["fbicoh"]) / 2])
        # ax.set_ylim([0, 20])

        ax.plot(
            [1, np.max(data_sub["fbicoh"]) / 2],
            [1, np.max(data_sub["fbicoh"]) / 2],
            "gray",
        )
        ax.plot(
            [np.max(data_sub["fbicoh"]) / 2, np.max(data_sub["fbicoh"])],
            [np.max(data_sub["fbicoh"]) / 2, 1],
            "gray",
        )
        # ax.set_title(sessions[i].sessinfo.session.sessionName)


# figure.savefig("phase_specific_slowgamma", __file__)


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


#%%* Multiple regression analysis on slow gamma power explained by variables such as theta-harmonic, theta-asymmetry, speed etc. Also comparing it with theta-harmonic being explained by similar variables
# region
gamma_expvar = pd.DataFrame()
harmonic_expvar = pd.DataFrame()

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
    thetaparams = sess.theta.getParams(lfpmaze)
    theta_trough = thetaparams.trough
    theta_amp = thetaparams.amplitude

    # --- calculating slow gamma parameters -------
    slowgammalfp = signal_process.filter_sig.bandpass(lfpmaze, lf=30, hf=58)
    hil_slowgamma = signal_process.hilbertfast(slowgammalfp)
    slow_gamma_amp = np.abs(hil_slowgamma) ** 2

    medgammalfp = signal_process.filter_sig.bandpass(lfpmaze, lf=62, hf=100)
    hil_medgamma = signal_process.hilbertfast(medgammalfp)
    med_gamma_amp = np.abs(hil_medgamma) ** 2

    fastgammalfp = signal_process.filter_sig.bandpass(lfpmaze, lf=100, hf=150)
    hil_fastgamma = signal_process.hilbertfast(fastgammalfp)
    fast_gamma_amp = np.abs(hil_fastgamma) ** 2

    # --- theta harmonic ----------
    theta_harmonic = signal_process.filter_sig.bandpass(lfpmaze, lf=10, hf=22)
    hil_theta_harmonic = signal_process.hilbertfast(theta_harmonic)
    theta_harmonic_amp = np.abs(hil_theta_harmonic) ** 2

    params = stats.binned_statistic(
        np.arange(len(lfpmaze)),
        [
            slow_gamma_amp,
            med_gamma_amp,
            fast_gamma_amp,
            theta_harmonic_amp,
            theta_amp,
            speed,
        ],
        bins=theta_trough,
    )[0]

    data = pd.DataFrame(
        {
            "slowgamma": params[0],
            "medgamma": params[1],
            "fastgamma": params[2],
            "thetaharmonic": params[3],
            "thetaPower": params[4],
            "speed": params[5],
            "asymm": thetaparams.asymmetry,
            "peaktrough": thetaparams.peaktrough,
        }
    )

    ind_var_gamma = data.columns.tolist()[3:]
    par_corr_stats_slowgamma = [
        data.partial_corr(
            y="slowgamma", x=var, covar=list(set(ind_var_gamma) - set([var]))
        )
        for var in ind_var_gamma
    ]

    par_corr_stats_medgamma = [
        data.partial_corr(
            y="medgamma", x=var, covar=list(set(ind_var_gamma) - set([var]))
        )
        for var in ind_var_gamma
    ]

    par_corr_stats_fastgamma = [
        data.partial_corr(
            y="fastgamma", x=var, covar=list(set(ind_var_gamma) - set([var]))
        )
        for var in ind_var_gamma
    ]

    ind_var_harmonic = data.columns.tolist()[4:]
    par_corr_stats_harmonic = [
        data.partial_corr(
            y="thetaharmonic", x=var, covar=list(set(ind_var_harmonic) - set([var]))
        )
        for var in ind_var_harmonic
    ]

    gamma_expvar = gamma_expvar.append(
        pd.DataFrame(
            {
                "slowgamma": [stat_.r2[0] * 100 for stat_ in par_corr_stats_slowgamma],
                "medgamma": [stat_.r2[0] * 100 for stat_ in par_corr_stats_medgamma],
                "fastgamma": [stat_.r2[0] * 100 for stat_ in par_corr_stats_fastgamma],
                "variables": ind_var_gamma,
            }
        )
    )
    # p_val_gamma = np.array([stat_["p-val"][0] for stat_ in par_corr_stats_gamma])

    harmonic_expvar = harmonic_expvar.append(
        pd.DataFrame(
            {
                "harmonic": [stat_.r2[0] * 100 for stat_ in par_corr_stats_harmonic],
                "variables": ind_var_harmonic,
            }
        )
    )
    # p_val_harmonic = np.array([stat_["p-val"][0] for stat_ in par_corr_stats_harmonic])


mean_gamma_expvar = gamma_expvar.groupby("variables").mean()
sem_gamma_expvar = gamma_expvar.groupby("variables").sem()
mean_harmonic_expvar = harmonic_expvar.groupby("variables").mean()
sem_harmonic_expvar = harmonic_expvar.groupby("variables").sem()

# ----- plotting ----------
figure = Fig()
fig, gs = figure.draw(grid=[4, 3])
ax1 = plt.subplot(gs[0, 0])
figure.panel_label(ax1, "c")
ax2 = plt.subplot(gs[0, 1], sharey=ax1)

mean_gamma_expvar.plot.bar(
    yerr=sem_gamma_expvar, ax=ax1, capsize=1.2, rot=90, edgecolor="k", width=0.8
)
# ax1.tick_params(axis="x", labelrotation=90)
ax1.set_ylabel("Explained variance (%)")
ax1.set_title("various gamma bands")
labels = mean_gamma_expvar.index.to_list()
ax1.set_xticks(range(5))
ax1.set_xticklabels(
    [
        "Sym. rise-decay",
        "Sym. peak-trough",
        "Speed",
        r"$\theta$ power",
        r"$\theta$ harmonic power",
    ]
)

ax1.legend(
    ["Slow-gamma (30-60 Hz)", "Medium-gamma (60-90 Hz)", "Fast-gamma (100-150 Hz)"]
)

mean_harmonic_expvar.plot.bar(
    yerr=sem_harmonic_expvar,
    ax=ax2,
    capsize=1.2,
    rot=90,
    edgecolor="k",
    width=0.4,
    legend=None,
)
# ax1.tick_params(axis="x", labelrotation=90)
ax2.set_xticks(range(4))
ax2.set_xticklabels(
    ["Sym. rise-decay", "Sym. peak-trough", "Speed", r"$\theta$ power",]
)

# ---- sanity plots for detection of theta params --------
# ax3 = plt.subplot(gs[1, :])
# theta_lfp = signal_process.filter_sig.bandpass(lfpmaze, lf=1, hf=25)
# ax3.plot(lfpmaze, "gray", alpha=0.3)
# ax3.plot(theta_lfp, "k")
# ax3.plot(theta_trough, theta_lfp[theta_trough], "|", markersize=30)
# ax3.plot(thetaparams.peak, theta_lfp[thetaparams.peak], "|", color="r", markersize=30)
# ax3.plot(
#     thetaparams.rise_mid,
#     theta_lfp[thetaparams.rise_mid],
#     "|",
#     color="gray",
#     markersize=30,
# )
# ax3.plot(
#     thetaparams.fall_mid,
#     theta_lfp[thetaparams.fall_mid],
#     "|",
#     color="magenta",
#     markersize=30,
# )
# ax3.plot(theta_trough, theta_lfp[theta_trough], "|", markersize=30)

# figure.savefig("gamma_bands_expvar", __file__)


# endregion

