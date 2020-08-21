#%%
import numpy as np
from callfunc import processData
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.stats as stats
import scipy.signal as sg
import pandas as pd
import seaborn as sns
import signal_process
import matplotlib as mpl
import warnings
import os
from plotUtil import savefig

warnings.simplefilter(action="default")


basePath = [
    "/data/Clustering/SleepDeprivation/RatJ/Day1/",
    "/data/Clustering/SleepDeprivation/RatK/Day1/",
    "/data/Clustering/SleepDeprivation/RatN/Day1/",
    "/data/Clustering/SleepDeprivation/RatJ/Day2/",
    "/data/Clustering/SleepDeprivation/RatK/Day2/",
    "/data/Clustering/SleepDeprivation/RatN/Day2/",
]


sessions = [processData(_) for _ in basePath]

plt.clf()
fig = plt.figure(1, figsize=(8.5, 11))
gs = gridspec.GridSpec(5, 4, figure=fig)
fig.subplots_adjust(hspace=0.5, wspace=0.3)
fig.suptitle("Sleep states related analysis")
titlesize = 8
panel_label = lambda ax, label: ax.text(
    x=-0.08,
    y=1.15,
    s=label,
    transform=ax.transAxes,
    fontsize=12,
    fontweight="bold",
    va="top",
    ha="right",
)

#%% Ripple probability vs delta amplitude divided into quantiles
# region
psth_all = []
for sub, sess in enumerate(sessions):
    sess.trange = np.array([])
    post = sess.epochs.post
    period = [post[0] + 5 * 3600, post[1]]
    psth = sess.eventpsth.hswa_ripple.compute(period=None)
    psth = pd.DataFrame(psth.T, columns=range(10))
    psth["time"] = np.linspace(-0.5, 0.5, 101) + 0.05
    psth = pd.melt(
        psth,
        id_vars=["time"],
        value_vars=range(10),
        var_name="quantile",
        value_name="counts",
    )
    psth["subname"] = [sess.sessinfo.session.sessionName] * len(psth)

    psth_all.append(psth)

psth_all = pd.concat(psth_all, ignore_index=True)
ax = fig.add_subplot(gs[0])
sns.lineplot(
    x="time",
    y="counts",
    hue="quantile",
    data=psth_all,
    palette="Greens",
    ci=None,
    # n_boot=1,
    legend=False,
)
ax.set_xlabel("Times from delta (s)")
ax.set_ylabel("Counts")
ax.set_title("Ripple probability", fontsize=titlesize)
panel_label(ax, "a")
# endregion


#%% Hourly delta ripple coupling over SD
# region
psth_all = []
for sub, sess in enumerate(sessions[:3]):
    sess.trange = np.array([])
    eegSrate = sess.recinfo.lfpSrate

    post = sess.epochs.post
    windows = np.arange(post[0], post[0] + 5 * 3600, 3600)

    psth = []

    for start in windows:
        period = [start, start + 3600]
        psth.append(
            sess.eventpsth.hswa_ripple.compute(period=period, nQuantiles=1).squeeze()
        )

    psth = np.asarray(psth)
    assert psth.shape[0] == len(windows)
    psth = pd.DataFrame(psth.T, columns=np.arange(1, 6))
    psth["time"] = np.linspace(-0.5, 0.5, 101) + 0.05
    psth = pd.melt(
        psth,
        id_vars=["time"],
        value_vars=np.arange(1, 6),
        var_name="window",
        value_name="counts",
    )
    psth["subname"] = [sess.sessinfo.session.sessionName] * len(psth)

    psth_all.append(psth)


psth_all = pd.concat(psth_all, ignore_index=True)
ax = fig.add_subplot(gs[1])
sns.lineplot(
    x="time",
    y="counts",
    hue="window",
    data=psth_all,
    palette="copper_r",
    ci=None,
    # n_boot=1,
    legend=False,
)
ax.set_xlabel("Times from delta (s)")
ax.set_ylabel("Counts")
ax.set_title("Ripple probability \n over SD", fontsize=titlesize)
# endregion


#%% Hourly delta-ripple coupling over recovery sleep
# region
psth_all = []
for sub, sess in enumerate(sessions[:3]):
    sess.trange = np.array([])
    eegSrate = sess.recinfo.lfpSrate

    post = sess.epochs.post
    windows = np.arange(post[0] + 5 * 3600, post[1], 3600)

    psth = []

    for start in windows[:5]:
        period = [start, start + 3600]
        psth.append(
            sess.eventpsth.hswa_ripple.compute(period=period, nQuantiles=1).squeeze()
        )

    psth = np.asarray(psth)
    psth = pd.DataFrame(psth.T, columns=np.arange(1, 6))
    psth["time"] = np.linspace(-0.5, 0.5, 101) + 0.05
    psth = pd.melt(
        psth,
        id_vars=["time"],
        value_vars=np.arange(1, 6),
        var_name="window",
        value_name="counts",
    )
    psth["subname"] = [sess.sessinfo.session.sessionName] * len(psth)

    psth_all.append(psth)


psth_all = pd.concat(psth_all, ignore_index=True)
ax = fig.add_subplot(gs[2])
sns.lineplot(
    x="time",
    y="counts",
    hue="window",
    data=psth_all,
    palette="copper_r",
    ci=None,
    # n_boot=1,
    legend=False,
)
ax.set_xlabel("Times from delta (s)")
ax.set_ylabel("Counts")
ax.set_title("Ripple probability \n during recovery sleep", fontsize=titlesize)
# endregion


#%% Hourly delta-ripple locking over the course of normal sleep (control sessions POST)
# region
psth_all = []
for sub, sess in enumerate(sessions[3:]):
    sess.trange = np.array([])
    eegSrate = sess.recinfo.lfpSrate

    post = sess.epochs.post
    windows = np.arange(post[0], post[1], 2 * 3600)

    psth = []

    for start in windows[:5]:
        period = [start, start + 2 * 3600]
        psth.append(
            sess.eventpsth.hswa_ripple.compute(period=period, nQuantiles=1).squeeze()
        )

    psth = np.asarray(psth)
    psth = pd.DataFrame(psth.T, columns=np.arange(1, 6))
    psth["time"] = np.linspace(-0.5, 0.5, 101) + 0.05
    psth = pd.melt(
        psth,
        id_vars=["time"],
        value_vars=np.arange(1, 6),
        var_name="window",
        value_name="counts",
    )
    psth["subname"] = [sess.sessinfo.session.sessionName] * len(psth)

    psth_all.append(psth)


psth_all = pd.concat(psth_all, ignore_index=True)
ax = fig.add_subplot(gs[3])
ax.clear()
sns.lineplot(
    x="time",
    y="counts",
    hue="window",
    data=psth_all,
    palette="copper_r",
    ci=None,
    # n_boot=1,
    legend=False,
)
ax.set_xlabel("Times from delta (s)")
ax.set_ylabel("Counts")
ax.set_title("Ripple probability \n over normal sleep", fontsize=titlesize)
# endregion

#%% hswa - spindle coactivity
# region
psth_all = []
for sub, sess in enumerate(sessions):
    sess.trange = np.array([])
    post = sess.epochs.post
    period = [post[0] + 5 * 3600, post[1]]
    psth = sess.eventpsth.hswa_spindle.compute(period=None, nQuantiles=10)
    psth = pd.DataFrame(psth.T, columns=range(10))
    psth["time"] = np.linspace(-0.5, 0.5, 101) + 0.05
    psth = pd.melt(
        psth,
        id_vars=["time"],
        value_vars=range(10),
        var_name="quantile",
        value_name="counts",
    )
    psth["subname"] = [sess.sessinfo.session.sessionName] * len(psth)

    psth_all.append(psth)

psth_all = pd.concat(psth_all, ignore_index=True)
ax = fig.add_subplot(gs[1, 0])
sns.lineplot(
    x="time",
    y="counts",
    hue="quantile",
    data=psth_all,
    palette="Greens",
    ci=None,
    # n_boot=1,
    legend=False,
)
ax.set_xlabel("Times from delta (s)")
ax.set_ylabel("Counts")
ax.set_title("Spindle probability", fontsize=titlesize)
panel_label(ax, "a")
# endregion


#%% Compare theta-gamma-phase coupling during REM
# region
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

    lfprem = []
    for epoch in rem.itertuples():
        lfprem.extend(binlfp(lfp, epoch.start, epoch.end))

    lfprem = np.asarray(lfprem)

    # xpac = p.filterfit(1250.0, lfprem, n_perm=20)
    theta_lfp = stats.zscore(signal_process.filter_sig.filter_theta(lfprem))
    hil_theta = signal_process.hilbertfast(theta_lfp)
    theta_amp = np.abs(hil_theta)
    theta_angle = np.angle(hil_theta, deg=True) + 180
    angle_bin = np.arange(0, 360, 20)
    bin_ind = np.digitize(theta_angle, bins=angle_bin)

    for band, (lfreq, hfreq) in enumerate(freqIntervals):
        gamma_lfp = stats.zscore(
            signal_process.filter_sig.filter_cust(lfprem, lf=lfreq, hf=hfreq)
        )

        hil_gamma = signal_process.hilbertfast(gamma_lfp)
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
        ax.set_xlabel("Degrees (from theta trough)")
        ax.set_ylabel("Amplitude")
        # p.comodulogram(
        #     xpac.mean(-1),
        #     title="Contour plot with 5 regions",
        #     cmap="Spectral_r",
        #     plotas="contour",
        #     ncontours=7,
        # )


# endregion

