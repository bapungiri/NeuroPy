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
for sub, sess in enumerate(sessions):

    sess.trange = np.array([])
    maze = sess.epochs.maze
    changrp = sess.recinfo.channelgroups[6]

    period = [maze[0], maze[0] + 3600]
    sess.lfpTheta = sess.utils.strong_theta_lfp(chans=changrp, period=period)

for sub, sess in enumerate(sessions):
    nChans = sess.lfpTheta.shape[0]
    nframes_2theta = 312
    n2theta = int(sess.lfpTheta.shape[1] / nframes_2theta)  # number of two theta cycles
    theta_last = signal_process.filter_sig.filter_cust(
        sess.lfpTheta[-1, :], lf=5, hf=12, ax=-1
    )
    peak = sg.find_peaks(theta_last)[0]
    peak = peak[np.where((peak > 1250) & (peak < len(theta_last) - 1250))[0]]

    avg_theta = np.zeros((16, 1250))
    for ind in peak:
        avg_theta = avg_theta + sess.lfpTheta[:, ind - 625 : ind + 625]

    avg_theta = avg_theta / len(peak)

    # lfptheta = sess.lfpTheta[:, : n2theta * nframes_2theta]
    # lfptheta = np.reshape(lfptheta, (nChans, nframes_2theta, n2theta), order="F").mean(
    #     axis=2
    # )
    # filter_theta = signal_process.filter_sig.filter_cust(lfptheta, lf=0.5, hf=20, ax=1)

    ycoord = np.arange(20, 17 * 20, 20)
    # xcoord = sess.recinfo.probemap()[:16] + 10
    # coords = tuple(zip(xcoord, ycoord))

    csd = signal_process.csdClassic(avg_theta, ycoord)
    plt.imshow(csd, aspect="auto")

    # csd_data = KCSD(xcoord.T, filter_theta)

    # from kcsd import KCSD2D

    # def do_kcsd(ele_pos, pots):
    #     h = 50.0  # distance between the electrode plane and the midslice
    #     sigma = 1.0  # S/m
    #     pots = pots.reshape((len(ele_pos), 1))  # first time point
    #     k = KCSD2D(
    #         ele_pos,
    #         pots,
    #         h=h,
    #         sigma=sigma,
    #         xmin=0.0,
    #         xmax=1.0,
    #         ymin=0.0,
    #         ymax=1.0,
    #         n_src_init=1000,
    #         src_type="gauss",
    #         R_init=1.0,
    #     )
    #     return k

    # k = KCSD2D(xcoord.T, filter_theta)
    # est_csd = k.values("CSD")

    # current = np.diff(np.diff(filter_theta, axis=0), axis=0)
    # ycoord = np.arange(0, 16 * 20, 20)
    # xcoord = np.arange(0, 312, 1)
    # f_csd = interp.interp2d(xcoord, ycoord, current)

    # sigarr = AnalogSignal(filter_theta.T, units="V", sampling_rate=1250 * pq.Hz)

    # x_req = np.arange(0, 312, 1)
    # y_req = np.arange(0, 14 * 20, 2)
    # csd = f_csd(x_req, y_req)


# endregion

