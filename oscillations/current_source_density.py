#%%
import warnings

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns

import signal_process
from callfunc import processData
from mathutil import threshPeriods
from plotUtil import Fig
from plotUtil import Colormap

# warnings.simplefilter(action="default")

# ===== Subjects =======
basePath = [
    # "/data/Clustering/SleepDeprivation/RatJ/Day1/",
    # "/data/Clustering/SleepDeprivation/RatK/Day1/",
    # "/data/Clustering/SleepDeprivation/RatN/Day1/",
    # "/data/Clustering/SleepDeprivation/RatJ/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day2/",
    # "/data/Clustering/SleepDeprivation/RatN/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day4/",
    "/data/Clustering/SleepDeprivation/RatN/Day4/",
    # "/data/Clustering/SleepDeprivation/RatA14d1LP/Rollipram/",
]


sessions = [processData(_) for _ in basePath]

#%% csd theta period during MAZE
# region

figure = Fig()
fig, gs = figure.draw(grid=[1, 8], wspace=0.3)
for sub, sess in enumerate(sessions):

    maze = sess.epochs.maze
    nShanks = sess.recinfo.nShanks

    for shank in range(nShanks):
        chans = sess.recinfo.goodchangrp[shank]
        csd = sess.theta.csd(period=maze, refchan=chans[4], chans=chans)

        ax = plt.subplot(gs[shank])
        csd.plot(ax=ax, plotLFP=True, cmap="jet_r")
        ax.set_title(f"Shank {shank+1}")

fig.suptitle("CSD Theta")

# figure.savefig("csd_theta_MAZE", __file__)

# endregion


#%% csd gamma period during MAZE
# region

for sub, sess in enumerate(sessions):

    maze = sess.epochs.maze
    channels = sess.recinfo.goodchans
    csd = sess.gamma.csd(
        period=maze, refchan=64, chans=channels, band=(25, 50), window=126
    )

figure = Fig()
fig, gs = figure.draw(grid=[4, 4])
ax = plt.subplot(gs[0])
csd.plot(smooth=2, ax=ax)
ax.set_title("CSD slow gamma")

figure.savefig("csd_slow gamma_MAZE", __file__)

# endregion


#%% CSD locked to ripple peak time
# region

for sub, sess in enumerate(sessions):

    sess.trange = np.array([])
    eegSrate = sess.recinfo.lfpSrate
    maze = sess.epochs.maze
    changrp = sess.recinfo.goodchangrp[0]
    peaktime = sess.ripple.events.peakSharpWave
    peakframe = (peaktime * eegSrate).astype(int)[2:4]

    # ------selecting time points around a subset of ripples ---------
    nfrm = 250
    nripples = 1
    rippleFrm = np.concatenate([np.arange(_ - nfrm, _ + nfrm) for _ in peakframe])
    lfpripple = sess.recinfo.geteeg(chans=changrp, frames=rippleFrm)
    lfpripple = lfpripple - np.mean(lfpripple)

    nChans = lfpripple.shape[0]
    filtrpl = signal_process.filter_sig.bandpass(lfpripple, lf=1, hf=50)
    # filtrpl = lfpripple
    analytic_signal = signal_process.hilbertfast(filtrpl, ax=-1)
    amplitude_envelope = np.abs(analytic_signal)
    mean_rpl = np.reshape(lfpripple, (nChans, 2 * nfrm, nripples * 2)).mean(axis=2)

    csd = signal_process.Csd(mean_rpl, coords=np.linspace(1, 100, 16))
    csd.classic()
    csd.plot()
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
    filteeg = signal_process.filter_sig.bandpass(eeg, lf=100, hf=150)
    hilberteeg = signal_process.hilbertfast(filteeg)
    amp_envlp_zsc = stats.zscore(np.abs(hilberteeg))


# endregion
