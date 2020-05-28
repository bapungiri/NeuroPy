import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.signal as sg
from matplotlib.gridspec import GridSpec
import seaborn as sns
from signal_process import filter_sig, hilbertfast
from statsmodels.tsa.stattools import grangercausalitytests
import matplotlib as mpl

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


# during Sleep deprivation

group = []
plt.clf()
fig = plt.figure(1, figsize=(1, 15))
gs = GridSpec(3, 1, figure=fig)
fig.subplots_adjust(hspace=0.5)

for sub, sess in enumerate(sessions):

    sess.trange = np.array([])
    tstart = sess.epochs.post[0]
    tend = sess.epochs.post[0] + 5 * 3600
    lfp, _, _ = sess.spindle.best_chan_lfp()
    t = np.linspace(0, len(lfp) / 1250, len(lfp))

    binwind = np.linspace(tstart, tend, 10)

    binlfp = lambda x, t1, t2: x[(t > t1) & (t < t2)]

    phase_amp = []
    for i in range(len(binwind) - 1):
        lfp_bin = binlfp(lfp, binwind[i], binwind[i + 1])

        theta_lfp = stats.zscore(filter_sig.filter_theta(lfp_bin))
        gamma_lfp = stats.zscore(filter_sig.filter_gamma(lfp_bin))

        hil_theta = hilbertfast(theta_lfp)
        hil_gamma = hilbertfast(gamma_lfp)

        theta_amp = np.abs(hil_theta)
        gamma_amp = np.abs(hil_gamma)

        theta_angle = np.angle(hil_theta, deg=True)
        angle_bin = np.arange(-180, 180, 20)
        bin_ind = np.digitize(theta_angle, bins=angle_bin)

        mean_amp = np.zeros(len(angle_bin) - 1)
        for i in range(1, len(angle_bin)):
            angle_ind = np.where(bin_ind == i)[0]
            mean_amp[i - 1] = gamma_amp[angle_ind].mean()

        # gamma_peaks, _ = sg.find_peaks(gamma_amp, height=5)
        # peak_angle, _ = np.histogram(
        #     theta_angle[gamma_peaks], bins=
        # )

        mean_amp_norm = mean_amp / np.sum(mean_amp)
        phase_amp.append(mean_amp_norm)

    phase_amp = np.asarray(phase_amp).T

    ax = fig.add_subplot(gs[sub, 0])
    # ax.imshow(phase, aspect="auto")
    cmap = mpl.cm.get_cmap("viridis")
    colmap = [cmap(x) for x in np.linspace(0, 1, phase_amp.shape[1])]
    for i in range(len(colmap)):
        ax.plot(angle_bin[:-1], phase_amp[:, i], color=colmap[i])
