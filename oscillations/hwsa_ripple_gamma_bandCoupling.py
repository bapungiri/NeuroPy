import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from matplotlib.gridspec import GridSpec
import seaborn as sns
from signal_process import spectrogramBands
from statsmodels.tsa.stattools import grangercausalitytests

from callfunc import processData

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
