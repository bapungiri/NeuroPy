#%%
import warnings

import elephant.current_source_density as csd2d
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import quantities as pq
import scipy.interpolate as interp
import scipy.signal as sg
import scipy.stats as stats
from kcsd.KCSD import KCSD
from neo.core import AnalogSignal
from scipy.ndimage import gaussian_filter, gaussian_filter1d

import signal_process
from callfunc import processData
from mathutil import threshPeriods
from plotUtil import Fig

warnings.simplefilter(action="default")

# ===== Subjects =======
basePath = [
    # "/data/Clustering/SleepDeprivation/RatJ/Day1/",
    # "/data/Clustering/SleepDeprivation/RatK/Day1/",
    # "/data/Clustering/SleepDeprivation/RatN/Day1/",
    # "/data/Clustering/SleepDeprivation/RatJ/Day2/",
    "/data/Clustering/SleepDeprivation/RatK/Day2/",
    # "/data/Clustering/SleepDeprivation/RatN/Day2/"
    # "/data/Clustering/SleepDeprivation/RatK/Day4/"
]


sessions = [processData(_) for _ in basePath]

#%% csd theta period during MAZE
# region

figure = Fig()
fig,gs = figure.draw(grid=[3,8])

# fig = plt.figure(1, figsize=(10, 15))
# gs = gridspec.GridSpec(1, 8, figure=fig)
# fig.subplots_adjust(hspace=0.3)
for sub, sess in enumerate(sessions):

    sess.trange = np.array([])
    maze = sess.epochs.maze
    changrp = np.concatenate(sess.recinfo.channelgroups[:8])

    period = [maze[0], maze[0] + 3600]
    sess.lfpTheta = sess.utils.strong_theta_lfp(chans=changrp, period=period)

for sub, sess in enumerate(sessions):
    nChans = sess.lfpTheta.shape[0]
    nframes_2theta = 312
    n2theta = int(sess.lfpTheta.shape[1] / nframes_2theta)  # number of two theta cycles
    theta_last = signal_process.filter_sig.filter_cust(
        sess.lfpTheta[16, :], lf=5, hf=12, ax=-1
    )
    peak = sg.find_peaks(theta_last)[0]
    peak = peak[np.where((peak > 1250) & (peak < len(theta_last) - 1250))[0]]

    avg_theta = np.zeros((nChans, 1250))
    for ind in peak:
        avg_theta = avg_theta + sess.lfpTheta[:, ind - 625 : ind + 625]

    avg_theta = avg_theta / len(peak)

    nshanks = sess.recinfo.nShanks
    changrp = np.concatenate(sess.recinfo.channelgroups[:8])
    for sh_id, shank in enumerate(sess.recinfo.channelgroups[:8]):

        chan_where = np.argwhere(np.isin(changrp, shank)).squeeze()
        theta_lfp = avg_theta[chan_where, :]
        badchans = sess.recinfo.badchans
        badchan_indx = np.argwhere(np.isin(shank, badchans))
        ycoord = np.arange(16 * 20, 0, -20)
        if badchan_indx.shape[0]:
            theta_lfp = np.delete(theta_lfp, badchan_indx, axis=0)
            ycoord = np.delete(ycoord, badchan_indx)

        # xcoord = np.asarray(sess.recinfo.probemap()[0][:16]) + 10
        # coords = np.vstack((xcoord, ycoord)).T

        # csd = signal_process.csdClassic(avg_theta, ycoord)
        # plt.imshow(csd, aspect="auto")

        sigarr = AnalogSignal(theta_lfp.T, units="uV", sampling_rate=1250 * pq.Hz)

        csd_data = csd2d.estimate_csd(
            sigarr, coords=ycoord.reshape(-1, 1) * pq.um, method="KCSD1D"
        )

        ypos = csd_data.annotations["x_coords"]
        t = np.linspace(-0.5, 0.5, 1250)
        theta_lfp = np.flipud(theta_lfp)

        ax = fig.add_subplot(gs[sh_id])
        im = ax.pcolorfast(t, ypos, np.asarray(csd_data).T, cmap="jet", zorder=1)
        ax2 = ax.twinx()
        ax2.plot(
            t,
            theta_lfp.T / 60000 + np.linspace(ypos[0], ypos[-1], theta_lfp.shape[0]),
            zorder=2,
            color="#616161",
        )
        ax.set_ylim([0, 0.35])
        ax2.axes.get_yaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_xlabel("Time (s)")
        ax.set_title(f"Shank{sh_id+1}")

        fig.colorbar(im, ax=ax, orientation="horizontal")
    fig.suptitle("Neptune Day1 - CSD (Strong theta during MAZE)")

figure.savefig('csd_theta_MAZE',__file__)

# endregion

#%% CSD locked to ripple peak time
# region
plt.close("all")

for sub, sess in enumerate(sessions):

    sess.trange = np.array([])
    eegSrate = sess.recinfo.lfpSrate
    maze = sess.epochs.maze
    changrp = np.concatenate(sess.recinfo.channelgroups[:8])
    ripples = sess.ripple.time
    peaktime = sess.ripple.peaktime
    peakshrpwv = sess.ripple.peakSharpWave
    peakframe = (peaktime * eegSrate).astype(int)
    startframe = (ripples[:, 0] * eegSrate).astype(int)

    # ------selecting time points around a subset of ripples ---------
    nfrm = 250
    nripples = 10
    rippleFrm = np.concatenate(
        [np.arange(_ - nfrm, _ + nfrm) for _ in peakshrpwv][:nripples]
    )
    lfpripple = sess.utils.geteeg(chans=changrp, frames=rippleFrm)
    lfpripple = lfpripple - np.mean(lfpripple)

    nChans = lfpripple.shape[0]
    filtrpl = signal_process.filter_sig.filter_cust(lfpripple, lf=2, hf=50)
    analytic_signal = signal_process.hilbertfast(filtrpl, ax=-1)
    amplitude_envelope = np.abs(analytic_signal)
    mean_rpl = np.reshape(lfpripple, (nChans, 2 * nfrm, nripples)).mean(axis=2)

    nshanks = sess.recinfo.nShanks
    changrp = np.concatenate(sess.recinfo.channelgroups[:8])
    sess.csd = []
    for sh_id, shank in enumerate(sess.recinfo.channelgroups[:8]):

        chan_where = np.argwhere(np.isin(changrp, shank)).squeeze()
        rpl_lfp = filtrpl[chan_where, :]
        badchans = sess.recinfo.badchans
        badchan_indx = np.argwhere(np.isin(shank, badchans))
        ycoord = np.arange(16 * 20, 0, -20)
        if badchan_indx.shape[0]:
            rpl_lfp = np.delete(rpl_lfp, badchan_indx, axis=0)
            ycoord = np.delete(ycoord, badchan_indx)
        ycoord = ycoord.reshape(-1, 1) * pq.um

        sigarr = AnalogSignal(rpl_lfp.T, units="uV", sampling_rate=1250 * pq.Hz)
        sess.csd.append(csd2d.estimate_csd(sigarr, coords=ycoord, method="KCSD1D"))

# -----Plotting---------
plt.clf()
fig = plt.figure(1, figsize=(10, 15))
gs = gridspec.GridSpec(1, 8, figure=fig)
fig.subplots_adjust(hspace=0.3)
for sub, sess in enumerate(sessions):

    for sh_id, shank in enumerate(sess.recinfo.channelgroups[:8]):
        csd_data = sess.csd[sh_id]
        ypos = csd_data.annotations["x_coords"]
        t = np.linspace(-nfrm / eegSrate, nfrm / eegSrate, 2 * nfrm * 10)
        chan_where = np.argwhere(np.isin(changrp, shank)).squeeze()
        rpl_lfp = mean_rpl[chan_where, :]
        # rpl_lfp = np.flipud(rpl_lfp)

        ax = fig.add_subplot(gs[sh_id])
        im = ax.pcolormesh(
            t, ypos, np.asarray(csd_data).T, cmap="jet", zorder=1, shading="nearest"
        )
        ax2 = ax.twinx()
        # ax2.plot(
        #     t,
        #     rpl_lfp.T / 60000 + np.linspace(ypos[0], ypos[-1], rpl_lfp.shape[0]),
        #     zorder=2,
        #     color="#616161",
        # )

        ax.set_ylim([0, 0.35])
        ax2.axes.get_yaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_xlabel("Time (s)")
        ax.set_title(f"Shank{sh_id+1}")

        fig.colorbar(im, ax=ax, orientation="horizontal")
    fig.suptitle("Neptune Day1 - CSD (around ripple peaktime)")
# endregion

#%% CSD locked to epsilon band peak (Fast gamma band > 100 Hz)
# region
"""
rationale : fast gamma captures spiking activity which should be localized around pyramidal layer as seen in the paper below.

Reference: belluscio et al. 2012
"""
plt.close("all")

for sub, sess in enumerate(sessions):

    sess.trange = np.array([])
    eegSrate = sess.recinfo.lfpSrate
    maze = sess.epochs.maze
    changrp = np.concatenate(sess.recinfo.channelgroups[:8])
    gammachan = 16
    eeg = sess.utils.geteeg(chans=gammachan, timeRange=maze)
    filteeg = signal_process.filter_sig.filter_cust(eeg, lf=100, hf=150)
    hilberteeg = signal_process.hilbertfast(filteeg)
    amp_envlp_zsc = stats.zscore(np.abs(hilberteeg))

    events = threshPeriods(
        amp_envlp_zsc, lowthresh=1, highthresh=2, minDistance=30, minDuration=50
    )

    filteeg = stats.zscore(filteeg)
    maxpos = [start + np.argmax(filteeg[start:end]) for (start, end) in events]

    eegall = sess.utils.geteeg(chans=changrp, timeRange=maze)
    filteegall = signal_process.filter_sig.filter_cust(eegall, lf=100, hf=150, ax=-1)

    #     # ------selecting time points around a subset of ripples ---------

    gamma_peak = [filteegall[:, frm - 25 : frm + 25] for frm in maxpos]
    gamma_avg = np.dstack(gamma_peak).mean(axis=2)

    nshanks = sess.recinfo.nShanks
    changrp = np.concatenate(sess.recinfo.channelgroups[:8])
    sess.csd = []
    for sh_id, shank in enumerate(sess.recinfo.channelgroups[:8]):

        chan_where = np.argwhere(np.isin(changrp, shank)).squeeze()
        rpl_lfp = gamma_avg[chan_where, :]
        badchans = sess.recinfo.badchans
        badchan_indx = np.argwhere(np.isin(shank, badchans))
        ycoord = np.arange(16 * 20, 0, -20)
        if badchan_indx.shape[0]:
            rpl_lfp = np.delete(rpl_lfp, badchan_indx, axis=0)
            ycoord = np.delete(ycoord, badchan_indx)
        ycoord = 
        .reshape(-1, 1) * pq.um

        sigarr = AnalogSignal(rpl_lfp.T, units="uV", sampling_rate=1250 * pq.Hz)
        sess.csd.append(csd2d.estimate_csd(sigarr, coords=ycoord, method="KCSD1D"))

# -----Plotting---------
plt.clf()
fig = plt.figure(1, figsize=(10, 15))
gs = gridspec.GridSpec(1, 8, figure=fig)
fig.subplots_adjust(hspace=0.3)
for sub, sess in enumerate(sessions):

    for sh_id, shank in enumerate(sess.recinfo.channelgroups[:8]):
        csd_data = sess.csd[sh_id]
        ypos = csd_data.annotations["x_coords"]
        t = np.linspace(-25 / eegSrate, 25 / eegSrate, 50) * 1000
        chan_where = np.argwhere(np.isin(changrp, shank)).squeeze()
        rpl_lfp = gamma_avg[chan_where, :]
        # rpl_lfp = np.flipud(rpl_lfp)

        ax = fig.add_subplot(gs[sh_id])
        im = ax.pcolormesh(
            t, ypos, np.asarray(csd_data).T, cmap="jet", zorder=1, shading="nearest"
        )
        ax2 = ax.twinx()
        ax2.plot(
            t,
            rpl_lfp.T / 10000 + np.linspace(ypos[0], ypos[-1], rpl_lfp.shape[0]),
            zorder=2,
            color="#616161",
        )

        ax.set_ylim([0, 0.35])
        ax2.axes.get_yaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_xlabel("Time (ms)")
        ax.set_title(f"Shank{sh_id+1}")

        fig.colorbar(im, ax=ax, orientation="horizontal")
    fig.suptitle("RatKDay2 - CSD (around peak fast gamma (100-150 Hz))")
# endregion
