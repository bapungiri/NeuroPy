#%%
import matplotlib.pyplot as plt
import scipy.signal as sg
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter
import scipy.interpolate as interp
import signal_process
import elephant.current_source_density as csd2d
from callfunc import processData
from neo.core import AnalogSignal
import quantities as pq
from kcsd.KCSD import KCSD

#%% Subjects
basePath = [
    # "/data/Clustering/SleepDeprivation/RatJ/Day1/",
    # "/data/Clustering/SleepDeprivation/RatK/Day1/",
    "/data/Clustering/SleepDeprivation/RatN/Day1/",
    # "/data/Clustering/SleepDeprivation/RatJ/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day2/",
    # "/data/Clustering/SleepDeprivation/RatN/Day2/"
    # "/data/Clustering/SleepDeprivation/RatK/Day4/"
]


sessions = [processData(_) for _ in basePath]

#%% csd theta period
# region
plt.close("all")
plt.clf()
fig = plt.figure(1, figsize=(10, 15))
gs = gridspec.GridSpec(1, 8, figure=fig)
fig.subplots_adjust(hspace=0.3)
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

    nshanks = sess.rec
    changrp = np.concatenate(sess.recinfo.channelgroups[:8])
    for sh_id, shank in enumerate(sess.recinfo.channelgroups[:8]):

        chan_where = np.argwhere(np.isin(changrp, shank)).squeeze()
        theta_lfp = avg_theta[chan_where, :]
        badchans = sess.recinfo.badchans
        badchan_indx = np.argwhere(np.isin(shank, badchans))
        ycoord = np.arange(20, 17 * 20, 20)
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

        fig.colorbar(im, ax=ax, orientation="horizontal")


# endregion

