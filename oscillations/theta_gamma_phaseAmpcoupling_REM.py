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
from tensorpac import Pac

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

# during REM sleep
plt.clf()
fig = plt.figure(1, figsize=(1, 15))
gs = GridSpec(4, 3, figure=fig)
fig.subplots_adjust(hspace=0.5)

colband = ["#CE93D8", "#1565C0", "#E65100"]
# p = Pac(idpac=(6, 3, 0), f_pha=(4, 10, 1, 1), f_amp=(30, 100, 5, 5))

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

    for epoch in rem.itertuples():
        lfprem.extend(binlfp(lfp, epoch.start, epoch.end))

    lfprem = np.asarray(lfprem)

    # xpac = p.filterfit(1250.0, lfprem, n_perm=20)
    theta_lfp = stats.zscore(filter_sig.filter_theta(lfprem))
    hil_theta = hilbertfast(theta_lfp)
    theta_amp = np.abs(hil_theta)
    theta_angle = np.angle(hil_theta, deg=True) + 180
    angle_bin = np.arange(0, 360, 20)
    bin_ind = np.digitize(theta_angle, bins=angle_bin)

    for band, (lfreq, hfreq) in enumerate(freqIntervals):
        gamma_lfp = stats.zscore(filter_sig.filter_cust(lfprem, lf=lfreq, hf=hfreq))

        hil_gamma = hilbertfast(gamma_lfp)
        gamma_amp = np.abs(hil_gamma)

        mean_amp = np.zeros(len(angle_bin) - 1)
        for i in range(1, len(angle_bin)):
            angle_ind = np.where(bin_ind == i)[0]
            mean_amp[i - 1] = gamma_amp[angle_ind].mean()

        mean_amp_norm = mean_amp / np.sum(mean_amp)

        ax = fig.add_subplot(gs[band + 1, plt_ind])
        ax.plot(
            angle_bin[:-1] + 10, mean_amp_norm, linestyle=lnstyle, color=colband[band]
        )
        ax.set_xlabel("Degree (from theta trough)")
        ax.set_ylabel("Amplitude")
        # p.comodulogram(
        #     xpac.mean(-1),
        #     title="Contour plot with 5 regions",
        #     cmap="Spectral_r",
        #     plotas="contour",
        #     ncontours=7,
        # )


# plt.clf()
ts = 2100
interval = np.arange(ts * 1250, (ts + 2) * 1250)
ax = fig.add_subplot(gs[0, :])
lfpsample = lfprem[interval]
troughs = sg.find_peaks(theta_angle[interval])[0]

for i in troughs:
    ax.plot([i, i], [-6000, 4000], "#EF9A9A")
ax.plot(lfpsample, "k")
ax.plot(filter_sig.filter_cust(lfpsample, lf=30, hf=50) - 2500, "#CE93D8")
ax.plot(filter_sig.filter_cust(lfpsample, lf=50, hf=90) - 4000, "#1565C0")
ax.plot(filter_sig.filter_cust(lfpsample, lf=100, hf=150) - 5500, "#E65100")
# ax.plot(filter_sig.filter_cust(lfpsample, lf=4, hf=10), "#E65100")
# ax.plot(theta_angle[interval], "#D500F9")
ax.text(0.02, 0.83, "Raw LFP", fontsize=10, transform=plt.gcf().transFigure)
ax.text(0.02, 0.79, "Low gamma (30-50 Hz)", fontsize=7, transform=plt.gcf().transFigure)
ax.text(
    0.02, 0.77, "High gamma (50-90 Hz)", fontsize=7, transform=plt.gcf().transFigure
)
ax.text(0.02, 0.75, "100-150 Hz", fontsize=7, transform=plt.gcf().transFigure)
ax.set_xlim([0, 2 * 1250])
ax.axis("off")

fig.suptitle("Theta phase - gamma amplitude modulation during REM")
