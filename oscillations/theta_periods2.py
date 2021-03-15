# %%
import warnings
from typing import Dict

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy
import pingouin as pg
import scipy.signal as sg
import scipy.stats as stats
import seaborn as sns
import signal_process
import subjects
from mathutil import threshPeriods
from plotUtil import Colormap, Fig
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from joblib import Parallel, delayed
import networkx as nx
from sklearn.cluster import spectral_clustering
import scipy.cluster.hierarchy as sch

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
        lfp,
        fs=1250,
        nperseg=window,
        noverlap=window / 6,
        detrend="linear",
    )
    noise = np.where(
        ((freq > 59) & (freq < 61)) | ((freq > 119) & (freq < 121)) | (freq > 220)
    )[0]
    freq = np.delete(freq, noise)
    Pxx = np.delete(Pxx, noise)

    return Pxx, freq


# endregion

#%% Example figure of power spectral density and changes w.r.t speed
# region

figure = Fig()
fig, gs = figure.draw(grid=(2, 2))
for sub, sess in enumerate(sessions):
    eegSrate = sess.recinfo.lfpSrate
    maze = sess.epochs.maze
    chan = sess.theta.bestchan
    eeg = sess.recinfo.geteeg(chans=chan, timeRange=maze)
    f, pxx = sg.welch(eeg, fs=eegSrate, nperseg=5 * 1250, noverlap=1250)

    ax = plt.subplot(gs[0])
    ax.plot(f, pxx)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")


# endregion

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

#%% Schematic --> Theta phase specific extraction method
# region

figure = Fig()
fig, gs = figure.draw(grid=(5, 3))
sessions = subjects.Of().ratNday4
fig.suptitle("Phase specfic extraction schematic")
for sub, sess in enumerate(sessions):
    eegSrate = sess.recinfo.lfpSrate
    maze = sess.epochs.maze
    thetachan = sess.theta.bestchan
    eeg = sess.recinfo.geteeg(chans=thetachan, timeRange=maze)
    strong_theta = stats.zscore(sess.theta.getstrongTheta(eeg)[0])
    rand_start = np.random.randint(0, len(strong_theta), 1)[0]
    theta_sample = strong_theta[rand_start : rand_start + 1 * eegSrate]
    thetaparams = sess.theta.getParams(theta_sample)
    gamma_lfp = signal_process.filter_sig.highpass(theta_sample, cutoff=25)

    # ----- dividing 360 degress into non-overlapping 5 bins ------------
    angle_bin = np.linspace(0, 360, 6)  # 5 bins so each bin=25ms
    angle_centers = angle_bin + np.diff(angle_bin).mean() / 2
    bin_ind = np.digitize(thetaparams.angle, bins=angle_bin)
    df = {}
    ax = plt.subplot(gs[0, :])
    cmap = mpl.cm.get_cmap("RdPu")
    for phase in range(1, len(angle_bin)):
        df[phase] = gamma_lfp[np.where(bin_ind == phase)[0]]

        ax.fill_between(
            np.arange(len(theta_sample)),
            np.min(theta_sample),
            theta_sample,
            where=(bin_ind == phase),
            # interpolate=False,
            color=cmap((phase + 1) / 10),
            # alpha=0.3,
            zorder=1,
        )
        # theta_atphase = theta_sample[np.where(bin_ind == phase)[0]]
        # ax.plot(theta_atphase)
    ax.plot(theta_sample, "k", zorder=2)
    ax.plot(thetaparams.lfp_filtered, "r", zorder=3)
    ax.plot(gamma_lfp - 3, color="#3b1641", zorder=3)
    ax.set_xlim([0, len(theta_sample)])
    ax.axis("off")

    axphase = plt.subplot(gs[1, :2])
    y_shift = 0.2
    for i in range(1, 6):
        axphase.plot(df[i] + y_shift, color=cmap((i + 1) / 10))
        axphase.axis("off")
        y_shift += 0.9
        axphase.set_ylim([-3.5, 4.8])
# figure.savefig("schematic_theta_phase_extraction", __file__)
# endregion

#%%* theta phase specific extraction of lfp during strong theta MAZE with different binning techiques
# region

figure = Fig()
fig, gs = figure.draw(grid=[4, 3], wspace=0.4)

for sub, sess in enumerate(sessions[7:8]):

    sess.trange = np.array([])
    eegSrate = sess.recinfo.lfpSrate
    maze = sess.epochs.maze

    lfpmaze = sess.recinfo.geteeg(chans=11, timeRange=maze)
    strong_theta = sess.theta.getstrongTheta(lfpmaze)[0]

    gamma_lfp = stats.zscore(
        signal_process.filter_sig.highpass(strong_theta, cutoff=25, order=3)
    )

    """
    phase specific extraction of highpass filtered strong theta periods (>25 Hz) and concatenating similar phases across multiple theta cycles
    """

    def getPxxData(**kwargs):

        gamma_bin, _, angle_centers = sess.theta.phase_specfic_extraction(
            strong_theta, gamma_lfp, **kwargs
        )
        df = pd.DataFrame()
        f_ = None
        for lfp, center in zip(gamma_bin, angle_centers):
            f_, pxx = sg.welch(lfp, nperseg=1250, noverlap=625, fs=1250)
            df[center] = np.log10(pxx)
        df.insert(0, "freq", f_)
        return df

    # ----- dividing 360 degress into multiple bins ------------
    binconfig = [[72, None], [40, None], [40, 5]]  # degree, degree
    binData = [getPxxData(binsize=wind, slideby=sld) for (wind, sld) in binconfig]

    bin_names = ["5bin", "9bin", "slide"]
    for i, df in enumerate(binData):
        ax = plt.subplot(gs[sub, i])
        data = df[df.freq < 200].set_index("freq")  # .transform(stats.zscore, axis=1)
        ax.pcolormesh(data.columns, data.index, data, cmap="jet", shading="auto")
        ax.set_xlabel(r"$\theta$ phase")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(bin_names[i])
        # ax.set_xticks([0, data.shape[1] // 2, data.shape[1]])
        # ax.set_xticklabels(["0", "180", "360"])
        # ax.locator_params(axis="x", nbins=4)

# axbin1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
# figure.savefig("phase_specific_slowgamma_openfield", __file__)

# endregion

#%% bicoherence in multiple channels from HIGH VELOCITY epochs on MAZE
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

#%% bicoherence at multiple channels during strong theta periods on MAZE
# region

data: Dict[str, np.array] = {}
for sub, sess in enumerate(sessions[8:9]):
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

    lfpmaze = sess.recinfo.geteeg(chans=chans2plot, timeRange=maze)
    lfpmaze = sess.artifact.removefrom(lfpmaze, timepoints=maze)
    strong_theta = sess.theta.getstrongTheta(lfpmaze)[0]

    # ------ psd calculation-----------
    f, pxx = sg.welch(
        strong_theta, fs=eegSrate, nperseg=4 * eegSrate, noverlap=eegSrate
    )
    # f_slow, pxx_slow = sg.welch(lfp_lowspd, fs=1250, nperseg=4 * 1250, noverlap=2 * 250)

    # ---- bicoherence calculation ----------
    # filtered_data = signal_process.filter_sig.bandpass(strong_theta, lf=1, hf=400)
    bicoh = signal_process.bicoherence(flow=1, fhigh=200)
    bicoh.compute(strong_theta)

    data[sub] = {
        "chans": chans2plot,
        # "fpxx_slow": f_slow,
        # "pxx_slow": pxx_slow,
        "fpxx": f,
        "pxx": pxx,
        "bicoh": bicoh,
    }


# ---- plotting ----------
figure = Fig()
cmap = Colormap().dynamic2()
for i in range(len(data)):
    data_sub = data[i]
    bicoh = data_sub["bicoh"]

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

        axbicoh = plt.subplot(gs[2 * chan + 1])
        bicoh.plot(index=chan, ax=axbicoh, cmap=cmap)


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
    specpower_sd = specpower.angles[
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

for sub, sess in enumerate(sessions[7:8]):
    sess.trange = np.array([])
    maze = sess.epochs.maze
    speed = sess.position.speed
    t_position = sess.position.t[1:]

    deadtime = sess.artifact.time

    lfpmaze = sess.recinfo.geteeg(sess.theta.bestchan, timeRange=maze)
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

    # ---- only selecting strong theta ------------
    lfpmaze, _, indices = sess.theta.getstrongTheta(lfpmaze)
    speed = speed[indices]

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
ax2 = plt.subplot(gs[0, 1])

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
    [
        "Sym. rise-decay",
        "Sym. peak-trough",
        "Speed",
        r"$\theta$ power",
    ]
)

figure.savefig("gamma_bands_expvar2", __file__)


# endregion

#%% Detect high gamma periods and then calculate at what theta phases they occurred
# region
gamma_at_thetaphase = pd.DataFrame()
for sub, sess in enumerate(sessions[9:10]):
    maze = sess.epochs.maze
    thetachan = sess.theta.bestchan
    lfp = sess.recinfo.geteeg(chans=thetachan, timeRange=maze)
    peakgamma = sess.gamma.getPeakIntervals(lfp, band=(25, 50))
    thetaparams = sess.theta.getParams(lfp)
    theta_phase = thetaparams.angle
    phase_bin = np.linspace(0, 360, 10)
    phase_centers = phase_bin[:-1] + np.diff(phase_bin).mean() / 2

    # ---- start of gamma w.r.t theta phase -------
    phase_hist_start = np.histogram(theta_phase[peakgamma[:, 0]], bins=phase_bin)[0]

    # ----- entire gamma event phase preference -------
    gamma_indices = np.concatenate([np.arange(beg, end) for (beg, end) in peakgamma])
    phase_hist_all = np.histogram(theta_phase[gamma_indices], bins=phase_bin)[0]

    gamma_at_thetaphase = gamma_at_thetaphase.append(
        pd.DataFrame(
            {
                "sub": sub,
                "phase": phase_centers,
                "start_phase": phase_hist_start / np.sum(phase_hist_start),
                "all_phase": phase_hist_all / np.sum(phase_hist_all),
            }
        )
    )

mean_group = gamma_at_thetaphase.groupby("phase").mean()
sem_group = gamma_at_thetaphase.groupby("phase").sem()

figure = Fig()
fig, gs = figure.draw(grid=(3, 3))
ax = plt.subplot(gs[0])
mean_group.plot(
    y="all_phase", yerr=sem_group.all_phase, legend=None, ax=ax, color="#DD2C00"
)
ax.set_title("Prefered phase for high gamma time points \n (0 degree = theta trough)")
ax.set_xlabel(r"$\theta$ Phase (degree)")
ax.set_ylabel(r"Normalized counts")


# endregion

#%% Detect high gamma periods for each channel individually and then calculate at what theta phases they occurred in the corresponding channel
# region
gamma_at_thetaphase = pd.DataFrame()
for sub, sess in enumerate(sessions):
    maze = sess.epochs.maze
    channels = sess.recinfo.goodchans
    nChans = len(channels)
    phase_bin = np.linspace(0, 360, 21)
    phase_centers = phase_bin[:-1] + np.diff(phase_bin).mean() / 2
    lfpmaze = sess.recinfo.geteeg(chans=channels, timeRange=maze)

    def theta_phase_pref(lfp):
        peakgamma = sess.gamma.getPeakIntervals(lfp, band=(100, 150))
        thetaparams = sess.theta.getParams(lfp, lowtheta=1, hightheta=25)
        theta_phase = thetaparams.angle

        # ----- entire gamma event phase preference -------
        gamma_indices = np.concatenate(
            [np.arange(beg, end) for (beg, end) in peakgamma]
        )
        phase_hist_all = np.histogram(theta_phase[gamma_indices], bins=phase_bin)[0]

        del thetaparams
        del peakgamma
        return phase_hist_all

    vals = Parallel(n_jobs=10)(delayed(theta_phase_pref)(lfp_) for lfp_ in lfpmaze)
    vals = stats.zscore(np.asarray(vals), axis=None)
    df = pd.DataFrame(vals.T, columns=channels)
    df["phase"] = phase_centers
    gamma_at_thetaphase = gamma_at_thetaphase.append(df)

group = gamma_at_thetaphase.set_index("phase")

figure = Fig()
fig, gs = figure.draw(num=1, grid=(4, 4))
ax = plt.subplot(gs[0])
sns.heatmap(data=group.T, ax=ax, cmap="Spectral_r", shading="gouraud", rasterized=True)
ax.set_title("fast gamma \n (0 degree = theta trough)")
ax.set_xlabel(r"$\theta$ Phase (degree)")
ax.set_ylabel("channels")

# figure.savefig("fastgamma_thetaphase", __file__)
# endregion

#%% Compare various methods of slow gamma detection during theta activity: wavelet, phase amplitude coupling, bicoherence
# region
"""For RatNDay4 (openfield), 
    chosen channel = 11 (pyramidal layer) 
"""
figure = Fig()
fig, gs = figure.draw(grid=(4, 3))
axwav = plt.subplot(gs[0])
axpac = plt.subplot(gs[1])
axbicoh = plt.subplot(gs[2])
for sub, sess in enumerate(sessions[7:8]):
    maze = sess.epochs.maze
    channels = 111
    phase_bin = np.linspace(0, 360, 21)
    phase_centers = phase_bin[:-1] + np.diff(phase_bin).mean() / 2
    lfpmaze = sess.recinfo.geteeg(chans=channels, timeRange=maze)
    strong_theta = sess.theta.getstrongTheta(lfpmaze)[0]

    # ----- wavelet power for gamma oscillations----------
    frgamma = np.arange(25, 150, 1)
    wavdec = signal_process.wavelet_decomp(strong_theta, freqs=frgamma)
    wav = wavdec.colgin2009()
    wav = stats.zscore(wav, axis=1)

    # ----segmenting gamma wavelet at theta phases ----------
    theta_params = sess.theta.getParams(strong_theta)
    bin_angle = np.linspace(0, 360, int(360 / 9) + 1)
    bin_ind = np.digitize(theta_params.angle, bin_angle)

    gamma_at_theta = pd.DataFrame()
    for i in np.unique(bin_ind):
        find_where = np.where(bin_ind == i)[0]
        gamma_at_theta[bin_angle[i - 1]] = np.mean(wav[:, find_where], axis=1)
    gamma_at_theta.insert(0, column="freq", value=frgamma)

    gamma_at_theta = gamma_at_theta.set_index("freq")
    sns.heatmap(gamma_at_theta, ax=axwav, cmap="Spectral_r")
    axwav.invert_yaxis()

    # ------ PAC (phase amplitude coupling)---------------
    pac = signal_process.PAC(fphase=(4, 12), famp=(25, 50), binsz=20)
    pac.compute(lfpmaze)
    pac.plot(ax=axpac, color="gray", edgecolor="k")
    axpac.set_xlabel(r"$\theta$ phase")
    axpac.set_xlabel("Gamma amplitude")
    axpac.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    # ------ bicoherence plot ------------
    colmap = Colormap().dynamic2()
    bicoh = signal_process.bicoherence(flow=1, fhigh=150)
    bicoh.compute(signal=lfpmaze)
    bicoh.plot(ax=axbicoh, cmap=colmap, smooth=3, vmax=0.05)

# figure.savefig("different_slow_gamma", __file__)

# endregion

#%% Wavelet --> theta-gamma coupling at multiple sites on a shank or across shanks
# region
"""For RatNDay4 (openfield), 
    chosen channel = 11 (pyramidal layer) 
"""
figure = Fig()
fig, gs = figure.draw(grid=(8, 8))
sessions = subjects.Of().ratNday4
for sub, sess in enumerate(sessions[7:8]):
    maze = sess.epochs.maze

    for shank in range(1):
        channels = sess.recinfo.goodchangrp[shank][::2]
        bin_angle = np.linspace(0, 360, int(360 / 9) + 1)
        phase_centers = bin_angle[:-1] + np.diff(bin_angle).mean() / 2
        lfpmaze = sess.recinfo.geteeg(chans=channels, timeRange=maze)

        frgamma = np.arange(25, 150, 1)

        def wavlet_(chan, lfp):
            strong_theta = sess.theta.getstrongTheta(lfp)[0]
            highpass_theta = signal_process.filter_sig.highpass(strong_theta, cutoff=25)

            # ----- wavelet power for gamma oscillations----------
            wavdec = signal_process.wavelet_decomp(highpass_theta, freqs=frgamma)
            wav = wavdec.colgin2009()
            wav = stats.zscore(wav, axis=1)

            # ----segmenting gamma wavelet at theta phases ----------
            theta_params = sess.theta.getParams(strong_theta)
            bin_ind = np.digitize(theta_params.angle, bin_angle)

            gamma_at_theta = pd.DataFrame()
            for i in np.unique(bin_ind):
                find_where = np.where(bin_ind == i)[0]
                gamma_at_theta[bin_angle[i - 1]] = np.mean(wav[:, find_where], axis=1)
            gamma_at_theta.insert(0, column="freq", value=frgamma)
            gamma_at_theta.insert(0, column="chan", value=chan)

            return gamma_at_theta

        gamma_all_chan = pd.concat(
            [wavlet_(chan, lfp) for (chan, lfp) in zip(channels, lfpmaze)]
        ).groupby("chan")

        for i, chan in enumerate(channels):
            spect = (
                gamma_all_chan.get_group(chan).drop(columns="chan").set_index("freq")
            )
            ax = plt.subplot(gs[i, shank])
            sns.heatmap(
                spect, ax=ax, cmap="jet", cbar=None, xticklabels=5, rasterized=True
            )
            # ax.pcolormesh(phase_centers, frgamma, spect, shading="auto")
            ax.set_title(f"channel = {chan}", loc="left")
            ax.invert_yaxis()
            if i < len(channels) - 1:
                ax.get_xaxis().set_visible([])
            if shank > 0:
                ax.get_yaxis().set_visible([])

    # figure.savefig(f"wavelet_slgamma", __file__)

# endregion

#%% Phase specific extraction --> theta-gamma coupling at multiple sites on a shank or across shanks
# region
figure = Fig()
fig, gs = figure.draw(num=1, grid=(8, 8))
sessions = subjects.Openfield().ratNday4
for sub, sess in enumerate(sessions):
    eegSrate = sess.recinfo.lfpSrate
    maze = sess.epochs.maze

    for shank in range(8):
        channels = sess.recinfo.goodchangrp[shank][::2]
        bin_angle = np.linspace(0, 360, int(360 / 9) + 1)
        phase_centers = bin_angle[:-1] + np.diff(bin_angle).mean() / 2
        lfpmaze = sess.recinfo.geteeg(chans=channels, timeRange=maze)

        frgamma = np.arange(25, 150, 1)

        def phase_(chan, lfp):
            strong_theta = sess.theta.getstrongTheta(lfp)[0]
            highpass_theta = signal_process.filter_sig.highpass(strong_theta, cutoff=25)
            gamma, _, angle = sess.theta.phase_specfic_extraction(
                strong_theta, highpass_theta, binsize=40, slideby=5
            )
            df = pd.DataFrame()
            f_ = None
            for gamma_bin, center in zip(gamma, angle):
                f_, pxx = sg.welch(gamma_bin, nperseg=1250, noverlap=625, fs=eegSrate)
                df[center] = pxx

            df.insert(0, "freq", f_)
            df.insert(0, "chan", chan)
            return df

        gamma_all_chan = pd.concat(
            [phase_(chan, lfp) for (chan, lfp) in zip(channels, lfpmaze)]
        ).groupby("chan")

        for i, chan in enumerate(channels):
            spect = (
                gamma_all_chan.get_group(chan).drop(columns="chan").set_index("freq")
            )
            # spect = spect[(spect.index > 25) & (spect.index < 150)]
            spect = spect[(spect.index > 25) & (spect.index < 150)]
            # spect = spect.transform(gaussian_filter1d, axis=1, sigma=2)
            ax = plt.subplot(gs[i, shank])
            sns.heatmap(
                spect,
                ax=ax,
                cmap="jet",
                cbar=None,
                xticklabels=10,
                rasterized=True,
                shading="gouraud",
            )
            # ax.pcolormesh(phase_centers, frgamma, spect, shading="auto")
            ax.set_title(f"channel = {chan}", loc="left")
            ax.invert_yaxis()
            if i < len(channels) - 1:
                ax.get_xaxis().set_visible([])
            if shank > 0:
                ax.get_yaxis().set_visible([])

            # ax.set_ylim([25, 150])


# figure.savefig(f"phase_specific_fourier_slgamma", __file__)

# endregion

#%% Theta phase specific extraction (MAZE) --> apply Colgin et al.2009 wavelet on extracted lfp
# region

figure = Fig()
fig, gs = figure.draw(grid=[4, 3])

sessions = subjects.Sd()
binData = None
for sub, sess in enumerate(sessions[7:8]):

    sess.trange = np.array([])
    eegSrate = sess.recinfo.lfpSrate
    maze = sess.epochs.maze

    lfpmaze = sess.recinfo.geteeg(chans=111, timeRange=maze)
    strong_theta = sess.theta.getstrongTheta(lfpmaze)[0]

    gamma_lfp = stats.zscore(
        signal_process.filter_sig.highpass(strong_theta, cutoff=25, order=3)
    )

    """
    phase specific extraction of highpass filtered strong theta periods (>25 Hz) and concatenating similar phases across multiple theta cycles
    """
    frgamma = np.arange(25, 150, 1)

    def getwavData(**kwargs):

        gamma_bin, _, angle_centers = sess.theta.phase_specfic_extraction(
            strong_theta, gamma_lfp, **kwargs
        )
        df = pd.DataFrame()
        df["freq"] = frgamma
        for lfp, center in zip(gamma_bin, angle_centers):
            wavdec = signal_process.wavelet_decomp(lfp, freqs=frgamma, sampfreq=1250)
            wav = wavdec.colgin2009()
            df[center] = np.mean(wav, axis=1)
        return df

    # ----- dividing 360 degress into multiple bins ------------
    binconfig = [[72, None], [40, None], [40, 5]]  # degree, degree
    binData = [getwavData(window=wind, slideby=sld) for (wind, sld) in binconfig]

    for i, df in enumerate(binData):
        ax = plt.subplot(gs[sub, i])
        data = df.set_index("freq")  # .transform(stats.zscore, axis=1)
        data = data.mul(frgamma, axis=0)
        sns.heatmap(data, ax=ax, cmap="Spectral_r", cbar=None, rasterized=True)
        # ax.set_xticks([0, data.shape[1] // 2, data.shape[1]])
        # ax.set_xticklabels(["0", "180", "360"])
        # ax.locator_params(axis="x", nbins=4)
        ax.set_xlabel(r"$\theta$ phase")
        ax.set_ylabel("Frequency (Hz)")
        ax.invert_yaxis()

figure.savefig("phase_specific_wavelet_scaled", __file__)
# endregion


#%% Compare before and after of theta phase specific extraction of various sub-gamma bands observing analyses e.g, wavelet, fourier, bicoherence
# region

figure = Fig()
fig, gs = figure.draw(grid=(4, 3))
axwav = plt.subplot(gs[0])
axpac = plt.subplot(gs[1])
axbicoh = plt.subplot(gs[2])

sessions = subjects.Nsd().ratNday2
for sub, sess in enumerate(sessions):
    maze = sess.epochs.maze
    channels = 111
    phase_bin = np.linspace(0, 360, 21)
    phase_centers = phase_bin[:-1] + np.diff(phase_bin).mean() / 2
    lfpmaze = sess.recinfo.geteeg(chans=channels, timeRange=maze)
    strong_theta = sess.theta.getstrongTheta(lfpmaze)[0]

    """ Before phase specific extraction """
    # ----- wavelet power for gamma oscillations----------
    frgamma = np.arange(25, 150, 1)
    wavdec = signal_process.wavelet_decomp(strong_theta, freqs=frgamma)
    wav = wavdec.colgin2009()
    wav = stats.zscore(wav, axis=1)

    theta_params = sess.theta.getParams(strong_theta)
    bin_angle = np.linspace(0, 360, int(360 / 9) + 1)
    bin_ind = np.digitize(theta_params.angle, bin_angle)

    gamma_at_theta = pd.DataFrame()
    for i in np.unique(bin_ind):
        find_where = np.where(bin_ind == i)[0]
        gamma_at_theta[bin_angle[i - 1]] = np.mean(wav[:, find_where], axis=1)
    gamma_at_theta.insert(0, column="freq", value=frgamma)

    gamma_at_theta = gamma_at_theta.set_index("freq")
    sns.heatmap(gamma_at_theta, ax=axwav, cmap="Spectral_r")
    axwav.invert_yaxis()

    # ------ PAC (phase amplitude coupling)---------------
    pac = signal_process.PAC(fphase=(4, 12), famp=(25, 50), binsz=20)
    pac.compute(lfpmaze)
    pac.plot(ax=axpac, color="gray", edgecolor="k")
    axpac.set_xlabel(r"$\theta$ phase")
    axpac.set_xlabel("Gamma amplitude")
    axpac.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    # ------ bicoherence plot ------------
    colmap = Colormap().dynamic2()
    bicoh = signal_process.bicoherence(flow=1, fhigh=150)
    bicoh.compute(signal=lfpmaze)
    bicoh.plot(ax=axbicoh, cmap=colmap, smooth=3, vmax=0.05)

    """After phase specific extraction """
    slgamma_highpass = signal_process.filter_sig.highpass(strong_theta, cutoff=25)
    gamma_bin, _, angle_centers = sess.theta.phase_specfic_extraction(
        strong_theta, slgamma_highpass, window=72, slideby=None
    )

    df = pd.DataFrame()
    df["freq"] = frgamma
    for lfp, center in zip(gamma_bin, angle_centers):
        wavdec = signal_process.wavelet_decomp(lfp, freqs=frgamma, sampfreq=1250)
        wav = wavdec.colgin2009()
        df[center] = np.mean(wav, axis=1)


# endregion


#%% Scratchpad --> Theta phase estimation using waveshape
# region
sessions = subjects.Sd().ratSday3
for sub, sess in enumerate(sessions):
    maze = sess.epochs.maze1
    chans = sess.recinfo.channelgroups[-1]
    lfp = np.asarray(sess.recinfo.geteeg(chans=67, timeRange=maze))
    corr_chans = np.corrcoef(lfp)
    corr_chans = np.where(corr_chans > 0.7, 1, 0)
    # plt.pcolormesh(np.array(chans), np.array(chans), corr_chans)
    # df = pd.DataFrame(corr_chans, columns=chans)
    # df.insert(0, "chans", chans)
    # df = df.set_index("chans")
    # sns.heatmap(df, annot=True)

    lfp_ca1 = np.median(lfp[:10, :], axis=0)
    strong_theta = sess.theta.getstrongTheta(lfp_ca1, lowthresh=0.5, highthresh=1)[0]

    theta_param1 = signal_process.ThetaParams(strong_theta, fs=1250, method="hilbert")
    theta_param2 = signal_process.ThetaParams(strong_theta, fs=1250, method="waveshape")

    # --- phase estimation by waveshape --------
    filt_lfp1 = signal_process.filter_sig.bandpass(strong_theta, lf=1, hf=80)
    filt_lfp1 = stats.zscore(filt_lfp1)
    peak = sg.find_peaks(filt_lfp1, height=0, distance=100, prominence=0.8)[0]
    trough = stats.binned_statistic(
        np.arange(len(filt_lfp1)), filt_lfp1, bins=peak, statistic=np.argmin
    )[0]
    trough = peak[:-1] + trough

    loc = np.concatenate((trough, peak))
    angles = np.concatenate((np.zeros(len(trough)), 180 * np.ones(len(peak))))
    sort_ind = np.argsort(loc)
    loc = loc[sort_ind]
    angles = angles[sort_ind]
    lfp_angle1 = np.interp(np.arange(len(strong_theta)), loc, angles)
    angle_descend = np.where(np.diff(lfp_angle1) < 0)[0]
    lfp_angle1[angle_descend] = -lfp_angle1[angle_descend] + 360

    # --- phase estimation by hilbert transform --------
    filt_lfp2 = signal_process.ThetaParams(strong_theta, lowtheta=1, hightheta=25)
    lfp_angle2 = filt_lfp2.angle

    a = np.concatenate([lfp_angle1, lfp_angle1 + 360, lfp_angle1, lfp_angle1 + 360])
    b = np.concatenate([lfp_angle2, lfp_angle2 + 360, lfp_angle2 + 360, lfp_angle2])

    plt.scatter(a, b, s=0.01)

    # for p in peak:
    #     plt.axvline(p, color="red")

    # for t in trough:
    #     plt.axvline(t, color='green')

    # filt_lfp = signal_process.filter_sig.bandpass(lfp, lf=1, hf=80)


# endregion

#%% Comparing gamma wavelet along theta using with phase estimation using waveshape and hilbert
# region
sessions = subjects.Sd().ratSday3
for sub, sess in enumerate(sessions):
    maze = sess.epochs.maze1
    chans = sess.recinfo.channelgroups[-1]
    lfp = np.asarray(sess.recinfo.geteeg(chans=chans, timeRange=maze))
    corr_chans = np.corrcoef(lfp)
    corr_chans = np.where(corr_chans > 0.7, 1, 0)

    lfp_ca1 = np.median(lfp[:8, :], axis=0)
    strong_theta = sess.theta.getstrongTheta(lfp_ca1, lowthresh=0.5, highthresh=1)[0]

    # --- phase estimation by waveshape --------

    theta_param1 = signal_process.ThetaParams(strong_theta, fs=1250, method="waveshape")
    lfp_angle1 = theta_param1.angle
    theta_param2 = signal_process.ThetaParams(strong_theta, fs=1250, method="hilbert")
    lfp_angle2 = theta_param2.angle

    frgamma = np.arange(25, 150)
    # ----- wavelet power for gamma oscillations----------
    wavdec = signal_process.wavelet_decomp(strong_theta, freqs=frgamma)
    wav = wavdec.colgin2009()
    wav = stats.zscore(wav, axis=1)

    # ----segmenting gamma wavelet at theta phases ----------
    bin_angle = np.linspace(0, 360, int(360 / 9) + 1)
    phase_centers = bin_angle[:-1] + np.diff(bin_angle).mean() / 2

    bin_ind1 = np.digitize(lfp_angle1, bin_angle)
    bin_ind2 = np.digitize(lfp_angle2, bin_angle)

    gamma_at_theta = pd.DataFrame()
    for i in np.unique(bin_ind1):
        find_where = np.where(bin_ind1 == i)[0]
        gamma_at_theta[bin_angle[i - 1]] = np.mean(wav[:, find_where], axis=1)
    gamma_at_theta.insert(0, column="freq", value=frgamma)
    gamma_at_theta.set_index("freq", inplace=True)

    gamma_at_theta2 = pd.DataFrame()
    for i in np.unique(bin_ind2):
        find_where = np.where(bin_ind2 == i)[0]
        gamma_at_theta2[bin_angle[i - 1]] = np.mean(wav[:, find_where], axis=1)
    gamma_at_theta2.insert(0, column="freq", value=frgamma)
    gamma_at_theta2.set_index("freq", inplace=True)

    # gamma_at_theta = gamma_at_theta.transform(stats.zscore, axis=1)
    # gamma_at_theta2 = gamma_at_theta2.transform(stats.zscore, axis=1)

    figure = Fig()
    fig, gs = figure.draw(num=1, grid=(1, 2))

    ax = plt.subplot(gs[0])
    sns.heatmap(gamma_at_theta, ax=ax, cmap="jet")
    ax.invert_yaxis()

    ax = plt.subplot(gs[1])
    sns.heatmap(gamma_at_theta2, ax=ax, cmap="jet")
    ax.invert_yaxis()


# endregion

#%% Gamma at theta phases on multiple shanks using (belluscio meethod) taking mean of correlated LFP
# region
figure = Fig()
fig, gs = figure.draw(num=1, grid=(4, 4))

# sessions = subjects.Sd().allsess[:3] + subjects.Nsd().allsess[:3]
# sessions = subjects.Of().ratNday4
sessions = subjects.Sd().ratNday1
for sub, sess in enumerate(sessions):
    maze = sess.epochs.maze
    eegSrate = sess.recinfo.lfpSrate
    ca1_chans = sess.ripple.bestchans
    nShanks = sess.recinfo.nShanks
    chan_grp = sess.recinfo.goodchangrp
    chan_grp = [_ for _ in chan_grp if _]

    gamma_all = []
    for chans in chan_grp:
        ca1_ind = np.intersect1d(chans, ca1_chans, return_indices=True)[1][0]
        chosen_chans = chans[: ca1_ind + 3]
        corr_chans = np.corrcoef(lfp)
        corr_chans = np.where(corr_chans > 0.7, 1, 0)

        lfp = np.asarray(sess.recinfo.geteeg(chans=chosen_chans, timeRange=maze))
        lfp_ca1 = np.median(lfp, axis=0)
        strong_theta = sess.theta.getstrongTheta(
            lfp_ca1, lowthresh=0.1, highthresh=0.5
        )[0]

        # --- phase estimation by waveshape --------

        theta_param = signal_process.ThetaParams(
            strong_theta, fs=1250, method="waveshape"
        )
        lfp_angle = theta_param.angle

        frgamma = np.arange(25, 150)
        # ----- wavelet power for gamma oscillations----------
        wavdec = signal_process.wavelet_decomp(
            strong_theta, freqs=frgamma, sampfreq=eegSrate
        )
        wav = wavdec.colgin2009()
        wav = stats.zscore(wav, axis=1)

        # ----segmenting gamma wavelet at theta phases ----------
        bin_angle = np.linspace(0, 360, int(360 / 9) + 1)
        phase_centers = bin_angle[:-1] + np.diff(bin_angle).mean() / 2

        bin_ind = np.digitize(lfp_angle, bin_angle)

        gamma_at_theta = pd.DataFrame()
        for i in np.unique(bin_ind):
            find_where = np.where(bin_ind == i)[0]
            gamma_at_theta[bin_angle[i - 1]] = np.mean(wav[:, find_where], axis=1)
        gamma_at_theta.insert(0, column="freq", value=frgamma)
        gamma_at_theta.set_index("freq", inplace=True)

        gamma_all.append(gamma_at_theta)

    for i, data in enumerate(gamma_all):
        ax = plt.subplot(gs[i])
        ax.contourf(
            phase_centers,
            frgamma,
            np.asarray(data),
            cmap="jet",
            levels=50,
            # origin="lower",
        )
        ax.set_xticks([0, 90, 180, 270, 360])
        ax.set_yticks(np.arange(25, 150, 30))
        ax.set_title(f"Shank {i+1}")


# figure.savefig("theta_gamma_belluscio")
# endregion

#%% Gamma frequency theta phase linear relationship
# region
sessions = subjects.Of().ratNday4
for sub, sess in enumerate(sessions):
    maze = sess.epochs.maze
    ca1_chans = sess.ripple.bestchans
    nShanks = sess.recinfo.nShanks
    chan_grp = sess.recinfo.goodchangrp
    chan_grp = [_ for _ in chan_grp if _]

    chosen_chans = []
    for chans in chan_grp:
        ca1_ind = np.intersect1d(chans, ca1_chans, return_indices=True)[1][0]
        chosen_chans.extend(chans[: ca1_ind + 1])

    lfp = np.asarray(sess.recinfo.geteeg(chans=chosen_chans, timeRange=maze))
    # corr_chans = np.corrcoef(lfp)
    # corr_chans = np.where(corr_chans > 0.7, 1, 0)

    lfp_ca1 = np.median(lfp, axis=0)
    strong_theta = sess.theta.getstrongTheta(lfp_ca1, lowthresh=0, highthresh=1)[0]

    # --- phase estimation by waveshape --------

    theta_param = signal_process.ThetaParams(strong_theta, fs=1250, method="waveshape")
    lfp_angle = theta_param.angle

    frgamma = np.arange(25, 150, 30)
    bin_angle = np.linspace(0, 360, int(360 / 9) + 1)

    hist_all = []
    for gamma in frgamma:
        gamma_lfp = signal_process.filter_sig.bandpass(
            strong_theta, lf=gamma, hf=gamma + 5
        )
        hilbert_sig = signal_process.hilbertfast(gamma_lfp)
        hilbert_amp = np.abs(hilbert_sig)
        periods = threshPeriods(
            stats.zscore(hilbert_amp),
            lowthresh=2,
            highthresh=2.1,
            minDistance=0,
            minDuration=10,
        )
        periods = np.ravel(periods)
        peak = stats.binned_statistic(
            np.arange(len(hilbert_amp)),
            hilbert_amp,
            bins=periods,
            statistic=np.argmax,
        )[0]
        peak = periods[:-1] + peak
        peak = peak[::2]
        gamma_phase = lfp_angle[peak.astype(int)]
        gamma_phase_hist = np.histogram(gamma_phase, bins=bin_angle)[0]
        hist_all.append(gamma_phase_hist)

    # frgamma = np.arange(25, 150)
    # # ----- wavelet power for gamma oscillations----------
    # wavdec = signal_process.wavelet_decomp(strong_theta, freqs=frgamma)
    # wav = wavdec.colgin2009()
    # wav = stats.zscore(wav, axis=1)

    # ----segmenting gamma wavelet at theta phases ----------
    # phase_centers = bin_angle[:-1] + np.diff(bin_angle).mean() / 2

    # bin_ind = np.digitize(lfp_angle, bin_angle)

    # gamma_at_theta = pd.DataFrame()
    # for i in np.unique(bin_ind):
    #     find_where = np.where(bin_ind == i)[0]
    #     gamma_at_theta[bin_angle[i - 1]] = np.mean(wav[:, find_where], axis=1)
    # gamma_at_theta.insert(0, column="freq", value=frgamma)
    # gamma_at_theta.set_index("freq", inplace=True)

    figure = Fig()
    fig, gs = figure.draw(num=1, grid=(4, 4))

    ax = plt.subplot(gs[0])
    # sns.heatmap(gamma_at_theta, ax=ax, cmap="jet")
    # ax.pcolormesh(bin_angle[1:], frgamma, np.asarray(gamma_at_theta), cmap="jet")
    # ax.contourf(
    #     phase_centers,
    #     frgamma,
    #     np.asarray(gamma_at_theta),
    #     cmap="jet",
    #     levels=50,
    #     # origin="lower",
    # )
    # ax.set_xlim([0, 360])
    ax.plot(np.asarray(hist_all).T)
    # ax.set_xticks([0, 90, 180, 270, 360])
    # ax.set_yticks(np.arange(25, 150, 30))

# endregion
