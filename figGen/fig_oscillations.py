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
gs = gridspec.GridSpec(5, 5, figure=fig)
fig.subplots_adjust(hspace=0.5, wspace=0.4)
fig.suptitle("Oscillation analysis")
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


#%% Ripple and delta waves density during Sleep deprivation
# region
data = []
for sub, sess in enumerate(sessions[:3]):
    sess.trange = np.array([])
    post = sess.epochs.post
    rpls = sess.ripple.time
    swa = sess.swa.time

    sd = np.linspace(post[0], post[0] + 5 * 3600, 6)
    rpls_sd = np.histogram(rpls, bins=sd)[0] / 3600
    swa_sd = np.histogram(swa, bins=sd)[0] / 60

    window = [f"ZT-{_}" for _ in range(len(sd) - 1)]
    data.append(pd.DataFrame({"hour": window, "ripple": rpls_sd, "delta": swa_sd}))

density = pd.concat(data)
axripple = plt.subplot(gs[0, 0])
sns.barplot(x="hour", y="ripple", data=density, ci="sd", ax=axripple)
axripple.set_ylabel("Ripples/s")
axripple.tick_params(axis="x", labelrotation=45)
panel_label(axripple, "a")

axdelta = plt.subplot(gs[1, 0])
sns.barplot(x="hour", y="delta", data=density, ci="sd")
axdelta.set_ylabel("delta/min")
axdelta.set_xlabel("")
axdelta.tick_params(axis="x", labelrotation=45)
# endregion

#%% Ripple, delta and spindle density during recovery sleep
# region
data = []
for sub, sess in enumerate(sessions[:3]):
    sess.trange = np.array([])
    post = sess.epochs.post
    rpls = sess.ripple.time
    swa = sess.swa.time
    spndl = sess.spindle.time

    states = sess.brainstates.states
    recslp_nrem = states.loc[
        (states.start > post[0] + 5 * 3600)
        & (states.name == "nrem")
        & (states.duration > 240),
        ["start", "end", "duration"],
    ]

    nrem_dur = np.asarray(recslp_nrem.duration)
    nrem_bins = recslp_nrem.to_numpy()[:, :2].flatten()
    rpls_nrem = np.histogram(rpls, bins=nrem_bins)[0][::2] / nrem_dur
    swa_nrem = np.histogram(swa, bins=nrem_bins)[0][::2] / nrem_dur
    spndl_nrem = np.histogram(spndl, bins=nrem_bins)[0][::2] / nrem_dur

    window = [nrem_ind + 1 for nrem_ind in range(len(nrem_dur))]
    data.append(
        pd.DataFrame(
            {
                "nrem": window,
                "ripple": rpls_nrem,
                "delta": swa_nrem,
                "spindle": spndl_nrem,
            }
        )
    )

density = pd.concat(data)
# order = np.unique(density.nrem)

axripple = plt.subplot(gs[0, 1:3])
axripple.clear()
sns.barplot(
    x="nrem", y="ripple", data=density, ci="sd", ax=axripple,
)
axripple.set_ylabel("Ripples / s")
axripple.tick_params(axis="x", labelrotation=45)
panel_label(axripple, "b")

axdelta = plt.subplot(gs[1, 1:3])
sns.barplot(x="nrem", y="delta", data=density, ci="sd")
axdelta.set_ylabel("delta / s")
axdelta.tick_params(axis="x", labelrotation=45)

axspndl = plt.subplot(gs[0, 3:])
sns.barplot(x="nrem", y="spindle", data=density, ci="sd")
axspndl.set_ylabel("spindle / s")
axspndl.tick_params(axis="x", labelrotation=45)
# endregion

#%% Hourly delta ripple coupling over SD and comparing with chance level
# region
psth_all = []
psthchance_all = []
for sub, sess in enumerate(sessions[:3]):
    sess.trange = np.array([])
    eegSrate = sess.recinfo.lfpSrate
    rpls = sess.ripple.time[:, 0]
    swa = sess.swa.time
    post = sess.epochs.post
    # windows = np.arange(post[0], post[0] + 5 * 3600, 3600)
    windows = [post[0], post[0] + 2 * 3600]
    npsth = len(windows)

    psth, psthchance = [], []

    for start in windows:
        period = [start, start + 2 * 3600]
        nrpls_window = len(rpls[(rpls > period[0]) & (rpls < period[1])])
        swa_window = swa[(swa > period[0]) & (swa < period[1])]
        rand_events = (
            np.diff(period)[0] * np.random.random_sample((nrpls_window,)) + period[0]
        )
        psth.append(
            sess.eventpsth.hswa_ripple.compute(period=period, nQuantiles=1).squeeze()
        )
        psthchance.append(sess.eventpsth.compute(swa, rand_events).squeeze())

    psth = np.asarray(psth)
    assert psth.shape[0] == len(windows)
    psth = pd.DataFrame(psth.T, columns=np.arange(1, npsth + 1))
    psth["time"] = np.linspace(-0.5, 0.5, 101) + 0.05
    psth = pd.melt(
        psth,
        id_vars=["time"],
        value_vars=np.arange(1, npsth + 1),
        var_name="window",
        value_name="counts",
    )
    psth["subname"] = [sess.sessinfo.session.sessionName] * len(psth)
    psth_all.append(psth)

    psthchance = np.asarray(psthchance)
    psthchance = pd.DataFrame(psthchance.T, columns=np.arange(1, npsth + 1))
    psthchance["time"] = np.linspace(-0.5, 0.5, 101) + 0.05
    psthchance = pd.melt(
        psthchance,
        id_vars=["time"],
        value_vars=np.arange(1, npsth + 1),
        var_name="window",
        value_name="counts",
    )
    psthchance["subname"] = [sess.sessinfo.session.sessionName] * len(psth)
    psthchance_all.append(psthchance)


psth_all = pd.concat(psth_all, ignore_index=True)
psthchance_all = pd.concat(psthchance_all, ignore_index=True)
axdel_rpl = fig.add_subplot(gs[2, 0])
# TODO chance level calculation
sns.lineplot(
    x="time",
    y="counts",
    hue="window",
    data=psth_all,
    # palette="copper_r",
    ci=48,
    n_boot=10,
    seed=10,
    legend=False,
    ax=axdel_rpl,
)
sns.lineplot(
    x="time",
    y="counts",
    hue="window",
    data=psthchance_all,
    # palette="copper_r",
    ci=48,
    n_boot=10,
    seed=10,
    legend=False,
    ax=axdel_rpl,
)
axdel_rpl.set_xlabel("Times from delta (s)")
axdel_rpl.set_ylabel("Counts")
axdel_rpl.set_title("Ripple probability over SD", fontsize=titlesize)
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
ax = fig.add_subplot(gs[2, 1])
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
ax = fig.add_subplot(gs[3, 1])
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

