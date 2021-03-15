# %%
import warnings
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import scipy.signal as sg
import scipy.stats as stats
import seaborn as sns
import signal_process
import subjects
from plotUtil import Colormap, Fig
from scipy.ndimage import gaussian_filter, gaussian_filter1d

# warnings.simplefilter(action="default")


#%% Ripple stats and examples
# region
figure = Fig()
fig, gs = figure.draw(num=1, grid=(2, 2))
sessions = subjects.Sd().ratNday1
for sub, sess in enumerate(sessions):
    maze = sess.epochs.maze
    sess.ripple.plot_summary(random=True)
# endregion

#%% Spindle stats and examples
# region
figure = Fig()
fig, gs = figure.draw(num=1, grid=(2, 2))
sessions = subjects.Sd().ratNday1
for sub, sess in enumerate(sessions):
    maze = sess.epochs.maze
    sess.spindle.plot()
# endregion

#%% Spectrogram, sleep fraction, ripple power SD, number of ripples
# region

figure = Fig()
fig, gs = figure.draw(num=1, grid=(3, 3), wspace=0.3)

# --- spectrogram -------
sessions = subjects.Sd().ratSday3 + subjects.Nsd().ratSday2
df = pd.DataFrame()

gs_spec = figure.subplot2grid(gs[0, :2], grid=(2, 1))
for sub, sess in enumerate(sessions):

    tag = sess.recinfo.animal.tag
    post = sess.epochs.post
    gs_ = figure.subplot2grid(gs_spec[sub], grid=(4, 1))
    axspec = fig.add_subplot(gs_[1:])
    chan = 77
    sess.viewdata.specgram(chan=chan, ax=axspec, window=10, overlap=2, period=post)
    axspec.text(10, 25, f"RatS-{tag.upper()}-POST", color="#fff")
    if sub == 0:
        axspec.axes.get_xaxis().set_visible(False)
        figure.panel_label(axspec, "a")
    axhypno = fig.add_subplot(gs_[0], sharex=axspec)
    sess.brainstates.hypnogram(ax=axhypno, tstart=post[0], unit="s")

# ----- sleep proportion ----------
gs_slp = figure.subplot2grid(gs[0, -1], grid=(2, 1))
sessions = subjects.Nsd() + subjects.Sd()
df_all = []
for sub, sess in enumerate(sessions):
    eegSrate = sess.recinfo.lfpSrate
    tag = sess.recinfo.animal.tag
    post = sess.epochs.post

    period = np.arange(post[0], post[1] - 900, 900)
    df = pd.DataFrame(columns=["nrem", "rem", "quiet", "active"])
    for st in period:
        states_prop = sess.brainstates.proportion(period=[st, st + 900])
        df = df.append(states_prop.T)

    df.fillna(0, inplace=True)
    df.insert(0, "window", np.arange(len(period)))
    df["grp"] = tag

    df_all.append(df)

df_all = pd.concat(df_all)
grp = df_all.groupby("grp")
sd_grp = grp.get_group("sd").groupby("window").mean()
nsd_grp = grp.get_group("nsd").groupby("window").mean()


ax = plt.subplot(gs_slp[0])
ax.clear()
sd_grp.plot.bar(
    ax=ax,
    y=["nrem", "rem"],
    stacked=True,
    width=1,
    color=sess.brainstates.colors,
    alpha=0.5,
    legend=None,
    rot=45,
)
ax.set_ylim(top=1)
ax.axes.get_xaxis().set_visible(False)
ax.set_ylabel("Sleep fraction")

ax = plt.subplot(gs_slp[1])
ax.clear()
nsd_grp.plot.bar(
    ax=ax,
    y=["nrem", "rem"],
    stacked=True,
    width=1,
    color=sess.brainstates.colors,
    alpha=0.5,
    legend=None,
    rot=45,
)
ax.set_ylim(top=1)
ax.axes.get_xaxis().set_visible(False)
ax.set_ylabel("sleep fraction")


# ---- ripple power spectrum ----------

sessions = subjects.Sd().allsess
df = pd.DataFrame()
for sub, sess in enumerate(sessions):
    post = sess.epochs.post
    sd = [post[0], post[0] + 5 * 3600]
    eegSrate = sess.recinfo.lfpSrate
    channel = sess.ripple.bestchans[3]
    post = sess.epochs.post
    lfp1st = sess.recinfo.geteeg(chans=channel, timeRange=[post[0], post[0] + 3600])
    lfp5th = sess.recinfo.geteeg(
        chans=channel, timeRange=[post[0] + 4 * 3600, post[0] + 5 * 3600]
    )

    psd = lambda sig: sg.welch(
        sig, fs=eegSrate, nperseg=5 * eegSrate, noverlap=2 * eegSrate
    )
    # multitaper = lambda sig: signal_process.mtspect(
    #     sig, fs=eegSrate, nperseg=10 * eegSrate, noverlap=5 * eegSrate
    # )

    _, psd1st = psd(lfp1st)
    f, psd5th = psd(lfp5th)

    # smoothing
    psd1st = gaussian_filter1d(psd1st, sigma=5)
    psd5th = gaussian_filter1d(psd5th, sigma=5)
    norm = np.sum(psd1st) + np.sum(psd5th)  # normalize

    df = df.append(pd.DataFrame({"freq": f, "psd": psd1st / norm, "hour": "first"}))
    df = df.append(pd.DataFrame({"freq": f, "psd": psd5th / norm, "hour": "last"}))


ax = plt.subplot(gs[1, 0])

colors = ["#f67e7e", "#d70427"]
for h, color in zip(["first", "last"], colors):
    df_ = df.groupby("hour").get_group(h).groupby("freq")
    f = df_.mean().index.values
    psd_mean = df_.mean().psd.values
    psd_sem = df_.sem(ddof=0).psd.values
    ax.fill_between(
        f,
        psd_mean - psd_sem,
        psd_mean + psd_sem,
        color=color,
        alpha=0.5,
        ec=None,
        label=h,
    )
    ax.plot(f, psd_mean, color=color)
    ax.set_xlim([120, 300])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim([0.5 * 10e-6, 0.5 * 10e-5])

ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Power (a.u.)")
ax.legend()
ax.set_title("Power in ripple band")

figure.panel_label(ax, "c")
# figure.savefig("ripple_power_sd", __file__)

# ----- number of ripples ------------------
data_sd, data_rs, data_nsd = [], [], []
sessions = subjects.Sd() + subjects.Nsd()
for sub, sess in enumerate(sessions):
    post = sess.epochs.post
    rpls = sess.ripple.events.start
    tag = sess.recinfo.animal.tag

    if tag == "sd":
        # -------sd period-----------
        sd = np.linspace(post[0], post[0] + 5 * 3600, 6)
        rpls_sd = np.histogram(rpls, bins=sd)[0] / 3600

        window = [f"ZT-{_}" for _ in range(len(sd) - 1)]
        data_sd.append(pd.DataFrame({"hour": window, "ripple": rpls_sd}))

        # -------- recovery sleep -------
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

        window = [nrem_ind + 1 for nrem_ind in range(len(nrem_dur))]
        data_rs.append(
            pd.DataFrame(
                {
                    "nrem": window,
                    "ripple": rpls_nrem,
                }
            )
        )

    else:
        # -------- recovery sleep -------
        states = sess.brainstates.states
        recslp_nrem = states.loc[
            (states.start > post[0])
            & (states.name == "nrem")
            & (states.duration > 240),
            ["start", "end", "duration"],
        ]

        nrem_dur = np.asarray(recslp_nrem.duration)
        nrem_bins = recslp_nrem.to_numpy()[:, :2].flatten()
        rpls_nrem = np.histogram(rpls, bins=nrem_bins)[0][::2] / nrem_dur

        window = [nrem_ind + 1 for nrem_ind in range(len(nrem_dur))]
        data_nsd.append(
            pd.DataFrame(
                {
                    "nrem": window,
                    "ripple": rpls_nrem,
                }
            )
        )


density_sd = pd.concat(data_sd)
density_rs = pd.concat(data_rs)
density_nsd = pd.concat(data_nsd)

gs_ = figure.subplot2grid(gs[1, 1:], grid=(2, 4), wspace=0.3, hspace=0.3)
axripple = plt.subplot(gs_[0, 0])
sns.barplot(
    x="hour", y="ripple", data=density_sd, ci="sd", ax=axripple, color="#d93a3a"
)

axripple.set_ylabel("Ripples/s")
axripple.tick_params(axis="x", labelrotation=45)
figure.panel_label(axripple, "d")

axrs = plt.subplot(gs_[0, 1:], sharey=axripple)
sns.barplot(x="nrem", y="ripple", data=density_rs, ci="sd", ax=axrs, color="#69c")
axrs.set_ylabel("")
axrs.tick_params(axis="x", labelrotation=45)

axnsd = plt.subplot(gs_[1, :], sharey=axripple)
sns.barplot(x="nrem", y="ripple", data=density_nsd, ci="sd", ax=axnsd, color="#69c")
axnsd.set_ylabel("Ripples/s")
axnsd.tick_params(axis="x", labelrotation=45)


# figure.savefig("proposal_sleep_ripple")

# endregion


#%% Explained variance
# region
figure = Fig()
fig, gs = figure.draw(num=1, grid=(3, 2))
sessions = (
    subjects.Nsd().ratSday2
    + subjects.Sd().ratSday3
    + subjects.Nsd().ratNday2
    + subjects.Sd().ratNday1
)

for sub, sess in enumerate(sessions):

    pre = sess.epochs.pre

    try:
        maze = sess.epochs.maze
    except:
        maze = sess.epochs.maze1
    # maze2 = sess.epochs.maze2

    post = sess.epochs.post
    # maze2 = sess.epochs.maze2
    # --- break region into periods --------
    bin1 = sess.utils.getinterval(pre, 2)
    bin2 = sess.utils.getinterval(post, 4)
    bins = bin1 + bin2
    sess.spikes.stability.firingRate(periods=bins)

    control = pre
    template = maze
    match = [post[0], post[1]]

    sess.replay.expvar.compute(
        template=template,
        match=match,
        control=control,
        slideby=300,
        cross_shanks=True,
    )
    print(sess.replay.expvar.npairs)

    axstate = figure.subplot2grid(gs[sub], grid=(4, 1))

    ax1 = fig.add_subplot(axstate[1:])
    sess.replay.expvar.plot(ax=ax1, tstart=post[0])
    ax1.set_xlim(left=0)
    ax1.tick_params(width=2)

    if sub == 3:
        ax1.set_ylim([0, 0.17])
    # ax1.spines["right"].set_visible("False")
    # ax1.spines["top"].set_visible("False")

    axhypno = fig.add_subplot(axstate[0], sharex=ax1)
    sess.brainstates.hypnogram(ax=axhypno, tstart=post[0], unit="h")
    # panel_label(axhypno, "a")
    # ax1.set_ylim([0, 11])


# figure.savefig("proposal_expvar")

# endregion

#%% Explained variance with same x-scale
# region
figure = Fig()
fig, gs = figure.draw(num=1, grid=(3, 2))
sessions = (
    subjects.Nsd().ratSday2
    + subjects.Sd().ratSday3
    + subjects.Nsd().ratNday2
    + subjects.Sd().ratNday1
)

for sub, sess in enumerate(sessions):

    pre = sess.epochs.pre

    try:
        maze = sess.epochs.maze
    except:
        maze = sess.epochs.maze1
    # maze2 = sess.epochs.maze2

    post = sess.epochs.post
    # maze2 = sess.epochs.maze2
    # --- break region into periods --------
    bin1 = sess.utils.getinterval(pre, 2)
    bin2 = sess.utils.getinterval(post, 4)
    bins = bin1 + bin2
    sess.spikes.stability.firingRate(periods=bins)

    control = pre
    template = maze
    match = [post[0], post[1]]

    sess.replay.expvar.compute(
        template=template,
        match=match,
        control=control,
        slideby=300,
        cross_shanks=True,
    )
    print(sess.replay.expvar.npairs)

    axstate = figure.subplot2grid(gs[sub], grid=(4, 1))

    ax1 = fig.add_subplot(axstate[1:])
    sess.replay.expvar.plot(ax=ax1, tstart=post[0])
    ax1.set_xlim(left=0)
    ax1.tick_params(width=2)

    if sub == 3:
        ax1.set_ylim([0, 0.17])
    # ax1.spines["right"].set_visible("False")
    # ax1.spines["top"].set_visible("False")

    axhypno = fig.add_subplot(axstate[0], sharex=ax1)
    sess.brainstates.hypnogram(ax=axhypno, tstart=post[0], unit="h")
    # panel_label(axhypno, "a")
    # ax1.set_ylim([0, 11])


# figure.savefig("proposal_expvar")

# endregion


#%% Place cells remapping
# region
figure = Fig()
fig, gs = figure.draw(num=1, grid=(3, 3))
sessions = subjects.Sd().ratSday3
for sub, sess in enumerate(sessions):
    # period = sess.epochs.maze1
    sess.placefield.pf1d.compute(track_name="maze1", run_dir="backward")

    ratemaps = np.asarray(sess.placefield.pf1d.ratemaps["ratemaps"])
    peak_frate = np.max(ratemaps, axis=1)
    good_cells = np.where(peak_frate > 1.5)[0]

    good_ratemaps = ratemaps[good_cells, :]
    cell_order = np.argsort(np.argmax(good_ratemaps, axis=1))
    good_cells = good_cells[cell_order]

    ax = plt.subplot(gs[:2, 0])
    sess.placefield.pf1d.plot(ax=ax, normalize=True, sortby=good_cells)
    # sess.placefield.pf1d.plot_raw(speed_thresh=True, subplots=None)
    # ax.set_title("MAZE")
    ax = plt.subplot(gs[:2, 1])
    sess.placefield.pf1d.compute(track_name="maze2", run_dir="backward")
    sess.placefield.pf1d.plot(
        ax=ax,
        normalize=True,
        sortby=good_cells,
    )
    # ax.set_title("Maze2")
    # sess.placefield.pf1d.plot_raw(speed_thresh=True, subplots=None)

# endregion


#%% Multiple regression theta parameters and slow gamm with wavelet analysis of slow gamma
# region
"""For RatNDay4 (openfield), 
    chosen channel = 11 (pyramidal layer) 
"""
figure = Fig()
fig, gs = figure.draw(num=1, grid=(5, 3), wspace=0.4)
sessions = subjects.Of().ratNday4
for sub, sess in enumerate(sessions):
    maze = sess.epochs.maze

    # ------ sharp wave ripple plot -------
    rpls = sess.ripple.events
    axripple = plt.subplot(gs[:3, 0])
    sess.ripple.plot_example(ax=axripple, ripple_indx=2457, shank_id=7, pad=0.6)

    gs_subplot = figure.subplot2grid(gs[:3, 1:], grid=(4, 2))
    for shank in [7]:
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
            ax = plt.subplot(gs_subplot[i])
            sns.heatmap(
                spect, ax=ax, cmap="jet", cbar=None, xticklabels=5, rasterized=True
            )
            # ax.pcolormesh(phase_centers, frgamma, spect, shading="auto")
            ax.set_title(f"channel = {chan}", loc="left")
            ax.invert_yaxis()

            if i < 6:
                ax.get_xaxis().set_visible([])
            if i % 2 != 0:
                ax.get_yaxis().set_visible([])

    # figure.savefig(f"wavelet_slgamma", __file__)


figure.savefig("proposal_ripple_wavelet_theta", __file__)


# endregion

#%% Schematic --> Theta phase specific extraction method and depthwise examples
# region

figure = Fig()
fig, gs = figure.draw(num=1, grid=(5, 3), wspace=0.4)
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
    # thetaparams = sess.theta.getParams(theta_sample)
    thetaparams = signal_process.ThetaParams(theta_sample)
    gamma_lfp = signal_process.filter_sig.highpass(theta_sample, cutoff=25)

    # ----- dividing 360 degress into non-overlapping 5 bins ------------
    angle_bin = np.linspace(0, 360, 6)  # 5 bins so each bin=25ms
    angle_centers = angle_bin + np.diff(angle_bin).mean() / 2
    bin_ind = np.digitize(thetaparams.angle, bins=angle_bin)
    df = {}
    ax = plt.subplot(gs[0, :])
    ax.clear()
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
    axphase.clear()
    y_shift = 0.25
    for i in range(1, 6):
        axphase.plot(df[i] + y_shift, color=cmap((i + 1) / 10))
        axphase.axis("off")
        y_shift += 0.95
        axphase.set_ylim([-3.5, 4.8])

    # ------ sharp wave ripple plot -------
    rpls = sess.ripple.events
    axripple = plt.subplot(gs[2:, 0])
    sess.ripple.plot_example(ax=axripple, ripple_indx=2457, shank_id=7, pad=0.6)
    # sess.ripple.export2Neuroscope()

    # ------ phase specific plot -------

    gs_subplot = figure.subplot2grid(gs[2:, 1:], grid=(4, 2))
    for shank in [7]:
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
            spect = spect[(spect.index > 25) & (spect.index < 150)].transform(
                stats.zscore, axis=1
            )

            spect = spect.transform(gaussian_filter1d, axis=1, sigma=3)
            spect = spect.transform(gaussian_filter1d, axis=0, sigma=3)
            ax = plt.subplot(gs_subplot[i])
            ax.clear()
            sns.heatmap(
                spect,
                ax=ax,
                cmap="jet",
                cbar=None,
                xticklabels=10,
                rasterized=True,
                shading=None,
            )
            # ax.pcolormesh(phase_centers, frgamma, spect, shading="auto")
            ax.set_title(f"channel = {chan}", loc="left")
            ax.invert_yaxis()
            if i < 6:
                ax.get_xaxis().set_visible([])
            if i % 2 != 0:
                ax.get_yaxis().set_visible([])

            # ax.set_ylim([25, 150])


# figure.savefig("proposal_theta_phase_extract", __file__)
# endregion


#%% Theta example
# region
figure = Fig()
fig, gs = figure.draw(num=1, grid=(2, 2))
sessions = subjects.Sd()
for sub, sess in enumerate(sessions):
    maze = sess.epochs.maze
# endregion

#%% Assembly activity during MAZE2 experience
# region
figure = Fig()
fig, gs = figure.draw(num=1, grid=(2, 2))

ax = plt.subplot(gs[1])
ax.set_xlabel("Reactivation strength")
ax.set_ylabel("Counts")
ax.set_yscale("log")
color = ["r", "k"]

gs_ = figure.subplot2grid(gs[0], grid=(2, 1))

sessions = subjects.Sd().ratSday3 + subjects.Nsd().ratSday2
for sub, sess in enumerate(sessions):
    maze1 = sess.epochs.maze1
    maze2 = sess.epochs.maze2

    sess.replay.assemblyICA.getAssemblies(period=maze1)
    activation_maze2, t_maze2 = sess.replay.assemblyICA.getActivation(period=maze2)
    # sess.replay.assemblyICA.plotActivation()

    bin_act = np.arange(0, 100, 2)
    hist_ = np.histogram(activation_maze2, bins=bin_act, density=True)[0]
    # cdf = np.cumsum(hist_)

    gs2 = figure.subplot2grid(gs_[sub], grid=(2, 1))

    for i in range(2):
        ax1 = plt.subplot(gs2[i])
        ax1.plot(activation_maze2[i], "k", alpha=0.5)
        ax1.set_xticks([])
        ax1.set_ylim([-20, 100])

    ax.plot(bin_act[1:], hist_, color=color[sub])

ax.legend(["SD", "NSD"])
# endregion

#%% Whavelet vs phase extraction comparison with one example figure only
# region
figure = Fig()
fig, gs = figure.draw(num=1, grid=(2, 2))
axwav = plt.subplot(gs[0])
axphase = plt.subplot(gs[1])

sessions = subjects.Of().ratNday4
for sub, sess in enumerate(sessions):
    eegSrate = sess.recinfo.lfpSrate
    maze = sess.epochs.maze
    channels = 111
    phase_bin = np.linspace(0, 360, 21)
    phase_centers = phase_bin[:-1] + np.diff(phase_bin).mean() / 2
    lfpmaze = sess.recinfo.geteeg(chans=channels, timeRange=maze)
    strong_theta = sess.theta.getstrongTheta(lfpmaze)[0]

    # ----- wavelet power for gamma oscillations----------
    frgamma = np.arange(20, 150, 2)
    wavdec = signal_process.wavelet_decomp(strong_theta, freqs=frgamma)
    wav = wavdec.colgin2009()
    # wav = wav - np.min(wav, axis=1, keepdims=True)
    # wav = wav / np.max(wav, axis=1, keepdims=True)
    wav = stats.zscore(wav, axis=1)

    theta_params = signal_process.ThetaParams(strong_theta)
    bin_angle = np.linspace(0, 360, int(360 / 9) + 1)
    bin_ind = np.digitize(theta_params.angle, bin_angle)

    gamma_at_theta = pd.DataFrame()
    for i in np.unique(bin_ind):
        find_where = np.where(bin_ind == i)[0]
        gamma_at_theta[bin_angle[i - 1]] = np.mean(wav[:, find_where], axis=1)
    gamma_at_theta.insert(0, column="freq", value=frgamma)

    gamma_at_theta = gamma_at_theta.set_index("freq")
    norm_gamma = stats.zscore(np.asarray(gamma_at_theta), axis=1)
    spect_wavelet = np.asarray(gamma_at_theta)
    # sns.heatmap(gamma_at_theta, ax=axwav, cmap="jet")
    axwav.pcolormesh(bin_angle, frgamma, spect_wavelet, cmap="jet", shading="auto")

    # ----- theta phase specific extraction ----------
    slgamma_highpass = signal_process.filter_sig.highpass(strong_theta, cutoff=25)
    gamma_bin, _, angle_centers = sess.theta.phase_specfic_extraction(
        strong_theta, slgamma_highpass, binsize=30, slideby=5
    )

    df = pd.DataFrame()
    for lfp, center in zip(gamma_bin, angle_centers):
        f_, pxx = sg.welch(lfp, nperseg=1250, noverlap=625, fs=eegSrate)
        df[center] = pxx
    df.insert(0, "freq", f_)
    df = df.set_index("freq")
    df = df[(df.index > 20) & (df.index < 150)].transform(stats.zscore, axis=1)

    freq_ = df.index.values
    spect = gaussian_filter(np.asarray(df), sigma=2)
    axphase.pcolormesh(angle_centers, freq_, spect, cmap="jet", shading="auto")


# figure.savefig("different_slow_gamma", __file__)


# endregion


#%% Power spectrum at varying speed
# region
figure = Fig()
fig, gs = figure.draw(num=1, grid=(2, 2), style="Pres")
sessions = subjects.Of().ratNday4

for sub, sess in enumerate(sessions):

    eegSrate = sess.recinfo.lfpSrate
    chan = 25
    maze = sess.epochs.maze
    mazepos = sess.tracks["maze"]
    speed = mazepos.speed
    speed = gaussian_filter1d(speed, sigma=10)

    mean_speed = stats.binned_statistic(
        mazepos.time, speed, statistic="mean", bins=np.arange(maze[0], maze[1], 1)
    )

    nQuantiles = 6
    quantiles = pd.qcut(mean_speed.statistic, nQuantiles, labels=False)

    colors = ["#757575", "#FF3D00"]
    for i, quant in enumerate([2, 5]):

        indx = np.where(quantiles == quant)[0]
        timepoints = mean_speed.bin_edges[indx]
        lfp_ind = np.concatenate(
            [
                np.arange(int(tstart * 1250), int((tstart + 1) * 1250))
                for tstart in timepoints
            ]
        )

        auc = []
        # for chan in goodchans:
        lfp = sess.recinfo.geteeg(chans=chan)
        lfp_quantile = lfp[lfp_ind]
        f, pxx = sg.welch(lfp_quantile, fs=1250, nperseg=2 * 1250, noverlap=1250)
        f_theta = np.where((f > 20) & (f < 100))[0]
        area_chan = np.trapz(y=pxx[f_theta], x=f[f_theta])
        auc.append(area_chan)
        ax = plt.subplot(gs[0])
        ax.plot(f, pxx, color=colors[i])
        ax.set_yscale("log")
        ax.set_xscale("log")

    ax.axvspan(25, 55, color="#FDD835", alpha=0.3)
    ax.set_xlim([3, 100])
    ax.set_ylim(bottom=100)
    ax.set_ylabel("Power")
    ax.set_xlabel("Frequency (Hz)")

    # for i in range(len(goodchans)):
    #     ax.plot(f, pxx[i, :])

# endregion
