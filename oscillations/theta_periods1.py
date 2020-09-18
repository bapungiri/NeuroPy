# %%

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as sg
import scipy.stats as stats
import seaborn as sns
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter, gaussian_filter1d
import ipywidgets as widgets
import random
from sklearn import linear_model
import statsmodels.api as sm
import pingouin as pg

from callfunc import processData
import signal_process
from mathutil import threshPeriods
import warnings

warnings.simplefilter(action="default")

#%% ====== functions needed for some computation ============
# region
def doWavelet(lfp, freqs, ncycles=3):
    wavdec = signal_process.wavelet_decomp(lfp, freqs=freqs)
    # wav = wavdec.cohen(ncycles=ncycles)
    wav = wavdec.colgin2009()

    wav = stats.zscore(wav)
    wav = gaussian_filter(wav, sigma=4)

    return wav


def getPxx(lfp):
    window = 5 * 1250

    freq, Pxx = sg.welch(
        lfp, fs=1250, nperseg=window, noverlap=window / 6, detrend="linear",
    )
    noise = np.where(
        ((freq > 59) & (freq < 61)) | ((freq > 119) & (freq < 121)) | (freq > 220)
    )[0]
    freq = np.delete(freq, noise)
    Pxx = np.delete(Pxx, noise)

    return Pxx, freq


# endregion

#%% Subjects to choose from
basePath = [
    "/data/Clustering/SleepDeprivation/RatJ/Day1/",
    "/data/Clustering/SleepDeprivation/RatK/Day1/",
    "/data/Clustering/SleepDeprivation/RatN/Day1/",
    "/data/Clustering/SleepDeprivation/RatJ/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day2/",
    "/data/Clustering/SleepDeprivation/RatN/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day4/"
]
sessions = [processData(_) for _ in basePath]


# %% Example plots for brief periods 10s on maze using Wavelets
# region
cmap = mpl.cm.get_cmap("hot_r")

plt.clf()
fig = plt.figure(1, figsize=(1, 15))
gs = gridspec.GridSpec(6, 4, figure=fig)
fig.subplots_adjust(hspace=0.5)
axgamma = fig.add_subplot(gs[0, :])
axtheta = fig.add_subplot(gs[1, :], sharex=axgamma)
axspeed = fig.add_subplot(gs[2, :], sharex=axgamma)
axlfp = fig.add_subplot(gs[3, :], sharex=axgamma)
axpos = fig.add_subplot(gs[4:6, 0])

colband = ["#CE93D8", "#1565C0", "#E65100"]

for sub, sess in enumerate(sessions):

    sess.trange = np.array([])
    posx = sess.position.x
    posy = sess.position.y
    post = sess.position.t
    maze = sess.epochs.maze

    lfp, _, _ = sess.spindle.best_chan_lfp()
    t = np.linspace(0, len(lfp) / 1250, len(lfp))

    tstart = maze[0] + 1500
    tend = maze[0] + 1510

    def allplts(tstart, tend):
        lfpmaze = lfp[(t > tstart) & (t < tend)]
        tmaze = np.linspace(tstart, tend, len(lfpmaze))
        posmazex = posx[(post > tstart) & (post < tend)]
        posmazey = posy[(post > tstart) & (post < tend)]
        postmaze = np.linspace(tstart, tend, len(posmazex))
        speed = np.sqrt(np.diff(posmazex) ** 2 + np.diff(posmazey) ** 2)
        speed = gaussian_filter1d(speed, sigma=10)

        axgamma.clear()
        frgamma = np.arange(25, 50, 1)
        wav = doWavelet(lfpmaze, freqs=frgamma, ncycles=7)
        axgamma.pcolorfast(tmaze, frgamma, wav, cmap="jet")
        axgamma.set_xlim([tstart, tend])
        axgamma.set_ylabel("Frequency")

        axtheta.clear()
        frtheta = np.arange(2, 20, 0.5)
        wav = doWavelet(lfpmaze, freqs=frtheta, ncycles=3)
        axtheta.pcolorfast(tmaze, frtheta, wav, cmap="jet")
        axtheta.set_ylabel("Amplitude")

        axpos.clear()
        colpos = [cmap(_ / len(posmazex)) for _ in range(len(posmazex))]
        # axpos.plot(postmaze, posmazex, "k")
        axpos.plot(posx, posy, color="#bfc0c0", zorder=1)
        # axpos.plot(postmaze, posmazey, "r")
        axpos.scatter(posmazex, posmazey, s=3, zorder=2, c=colpos)
        axpos.axis("off")

        axspeed.clear()
        # axspeed.plot(postmaze[:-1], np.abs(np.diff(posmazex)), "k")
        # axspeed.plot(postmaze[:-1], np.abs(np.diff(posmazey)), "r")
        axspeed.plot(postmaze[:-1], speed, "k")
        axspeed.set_ylabel("Speed (a.u.)")

        axlfp.clear()
        axlfp.plot(tmaze, lfpmaze, "k")
        axlfp.set_xlabel("Time (s)")
        axlfp.set_ylabel("Amplitude")

    allplts(tstart, tend)

    # ax = fig.add_subplot(gs[5, 1:])
    # axcolor = "lightgoldenrodyellow"
    # # axfreq = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    # timetoPlot = Slider(
    #     ax, "Time", maze[0], maze[1], valinit=tstart, valstep=10, fc="gray"
    # )
    @widgets.interact(time=(maze[0], maze[1], 10))
    def update(time=tstart):
        # tnow = timetoPlot.val
        allplts(time - 5, time + 5)
        # l.set_ydata(amp * np.sin(2 * np.pi * freq * t))
        # fig.canvas.draw_idle()

    # timetoPlot.on_changed(update)
    # samp.on_changed(update)
# endregion

# %% This detects strong theta periods within the maze exploration using Wavelets
# region
cmap = mpl.cm.get_cmap("hot_r")

for sub, sess in enumerate(sessions):

    sess.trange = np.array([])
    posx = sess.position.x
    posy = sess.position.y
    post = sess.position.t
    maze = sess.epochs.maze

    lfp, _, _ = sess.spindle.best_chan_lfp()
    t = np.linspace(0, len(lfp) / 1250, len(lfp))

    tstart = maze[0]
    tend = maze[1]

    lfpmaze = lfp[(t > tstart) & (t < tend)]
    tmaze = np.linspace(tstart, tend, len(lfpmaze))
    posmazex = posx[(post > tstart) & (post < tend)]
    posmazey = posy[(post > tstart) & (post < tend)]
    postmaze = np.linspace(tstart, tend, len(posmazex))
    speed = np.sqrt(np.diff(posmazex) ** 2 + np.diff(posmazey) ** 2)
    speed = stats.zscore(gaussian_filter1d(speed, sigma=10))

    # frgamma = np.arange(25, 50, 1)
    # wavgamma = doWavelet(lfpmaze, freqs=frgamma, ncycles=7)

    frtheta = np.arange(5, 12, 0.5)
    wavdec = signal_process.wavelet_decomp(lfpmaze, freqs=frtheta)
    wav = wavdec.cohen(ncycles=7)
    # wavtheta = doWavelet(lfpmaze, freqs=frtheta, ncycles=3)

    sum_theta = gaussian_filter1d(np.sum(wav, axis=0), sigma=10)
    zsc_theta = stats.zscore(sum_theta)
    thetaevents = threshPeriods(
        zsc_theta, lowthresh=0, highthresh=1.5, minDistance=625, minDuration=2 * 1250
    )
    thetaevents = thetaevents / 1250 + tstart

    plt.clf()
    fig = plt.figure(1, figsize=(10, 15))
    outer = gridspec.GridSpec(2, 5, figure=fig)
    fig.subplots_adjust(hspace=0.2)

    def ex_plts(tstart, tend, pltind):
        inner = gridspec.GridSpecFromSubplotSpec(
            5, 3, subplot_spec=outer[pltind], wspace=0.1, hspace=0.3
        )

        axgamma = fig.add_subplot(inner[0, :])
        axtheta = fig.add_subplot(inner[1, :], sharex=axgamma)
        axspeed = fig.add_subplot(inner[2, :], sharex=axgamma)
        axlfp = fig.add_subplot(inner[3, :], sharex=axgamma)
        axpos = fig.add_subplot(inner[4, 0])

        lfpevent = lfp[(t > tstart) & (t < tend)]
        tevent = np.linspace(tstart, tend, len(lfpevent)) - tstart
        poseventx = posx[(post > tstart) & (post < tend)]
        poseventy = posy[(post > tstart) & (post < tend)]
        postevent = np.linspace(tstart, tend, len(poseventx)) - tstart
        speedevent = np.sqrt(np.diff(poseventx) ** 2 + np.diff(poseventy) ** 2)
        speedevent = gaussian_filter1d(speedevent, sigma=10)

        frgamma = np.arange(25, 50, 1)
        wav = doWavelet(lfpevent, freqs=frgamma, ncycles=7)
        axgamma.pcolorfast(tevent, frgamma, wav, cmap="jet")
        axgamma.set_xlim([tevent[0], tevent[-1]])
        axgamma.set_ylabel("Frequency (Hz)")
        axgamma.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

        frtheta = np.arange(2, 20, 0.5)
        wav = doWavelet(lfpevent, freqs=frtheta, ncycles=7)
        axtheta.pcolorfast(tevent, frtheta, wav, cmap="jet")
        axtheta.set_ylabel("Frequency (Hz)")
        axtheta.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

        axspeed.plot(postevent[:-1], speedevent, "k")
        axspeed.set_ylabel("Speed (a.u.)")
        axspeed.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        axspeed.set_ylim([0, 5])

        axlfp.plot(tevent, stats.zscore(lfpevent), "k")
        axlfp.set_xlabel("Time (s)")
        axlfp.set_ylabel("Amplitude")

        colpos = [cmap(_ / len(poseventx)) for _ in range(len(poseventx))]
        axpos.plot(posmazey, posmazex, color="#bfc0c0", zorder=1)
        axpos.scatter(poseventy, poseventx, s=3, zorder=2, c=colpos)
        axpos.axis("off")

    for plt_id, evt in enumerate(random.sample(range(thetaevents.shape[0]), 10)):
        ex_plts(thetaevents[evt, 0], thetaevents[evt, 1], plt_id)

    # @widgets.interact(time=(maze[0], maze[1], 10))
    # def updateExamples(time=tstart):
    #     # tnow = timetoPlot.val
    #     allplts(time - 5, time + 5)
    fig.suptitle("Example periods of strong theta in RatN Day1")
# endregion

# %% Detects strong theta within maze and averages spectrogram around theta cycles using Wavelets to look for gamma modulation (25-150 Hz)
# region
inner = gridspec.GridSpecFromSubplotSpec(
    2, 1, subplot_spec=gs[0, 0], wspace=0.1, hspace=0.1
)
cmap = mpl.cm.get_cmap("hot_r")
gamma_at_theta_all = pd.DataFrame()
for sub, sess in enumerate(sessions):

    sess.trange = np.array([])
    eegSrate = sess.recinfo.lfpSrate
    maze = sess.epochs.maze

    # --- get maze lfp ---------
    lfpmaze = sess.utils.geteeg(sess.theta.bestchan, timeRange=maze)
    tmaze = np.linspace(maze[0], maze[1], len(lfpmaze))

    # ---- filtering --> zscore --> threshold --> strong theta periods ----
    thetalfp = signal_process.filter_sig.bandpass(lfpmaze, lf=4, hf=10)
    hil_theta = signal_process.hilbertfast(thetalfp)
    theta_amp = np.abs(hil_theta)

    zsc_theta = stats.zscore(theta_amp)
    thetaevents = threshPeriods(
        zsc_theta, lowthresh=0, highthresh=0.5, minDistance=300, minDuration=1250
    )

    strong_theta, theta_indices = [], []
    for (beg, end) in thetaevents:
        strong_theta.extend(lfpmaze[beg:end])
        theta_indices.extend(np.arange(beg, end))
    strong_theta = np.asarray(strong_theta)
    theta_indices = np.asarray(theta_indices)
    non_theta = np.delete(lfpmaze, theta_indices)

    # ----- wavelet power for gamma oscillations----------
    frgamma = np.arange(25, 150, 1)
    # frgamma = np.linspace(25, 150, 1)
    wavdec = signal_process.wavelet_decomp(strong_theta, freqs=frgamma)
    wav = wavdec.colgin2009()
    # wav = wavdec.cohen(ncycles=7)
    wav = stats.zscore(wav, axis=1)

    # ----segmenting gamma wavelet at theta phases ----------
    theta_params = sess.theta.getParams(strong_theta)
    bin_angle = np.linspace(0, 360, int(360 / 9) + 1)
    bin_ind = np.digitize(theta_params.angle, bin_angle)

    gamma_at_theta = pd.DataFrame()
    for i in np.unique(bin_ind):
        find_where = np.where(bin_ind == i)[0]
        gamma_at_theta[bin_angle[i - 1]] = np.mean(wav[:, find_where], axis=1)
    gamma_at_theta.insert(0, column="freq", value=frgamma)

    # ---- appending for all subjects ------------
    gamma_at_theta_all = gamma_at_theta_all.append(gamma_at_theta)

ax = fig.add_subplot(inner[:, :])
mean_gamma = (
    gamma_at_theta_all.groupby(by="freq").mean().transform(stats.zscore, axis=1)
)

sns.heatmap(mean_gamma, robust=True, cmap="Spectral_r", shading="gouraud", ax=ax)
ax.invert_yaxis()
ax.set_xlabel(r"$\theta$ phase")
ax.set_ylabel("Frequency (Hz)")

# endregion

#%% Theta within REM sleep and averages spectrogram around theta cycles usign wavelets to look for gamma modulation (25-150 Hz)
# region
plt.clf()
fig = plt.figure(1, figsize=(10, 15))
gs = gridspec.GridSpec(1, 6, figure=fig)
fig.subplots_adjust(hspace=0.3)
for sub, sess in enumerate(sessions):

    sess.trange = np.array([])
    eegSrate = sess.recinfo.lfpSrate
    states = sess.brainstates.states
    rems = states.loc[states["name"] == "rem"]

    lfp = sess.theta.getBestChanlfp()
    if sub in [1, 4]:
        lfp = sess.utils.geteeg(chans=50)

    rem_frames = []
    for rem in rems.itertuples():
        rem_frames.extend(
            list(range(int(rem.start * eegSrate), int(rem.end * eegSrate)))
        )
    rem_theta = lfp[rem_frames]

    # ----- wavelet power for gamma oscillations----------
    frgamma = np.arange(25, 150, 1)
    # frgamma = np.linspace(25, 150, 1)
    wavdec = signal_process.wavelet_decomp(rem_theta, freqs=frgamma)
    wav = wavdec.colgin2009()
    # wav = wavdec.cohen(ncycles=7)
    wav = stats.zscore(wav, axis=1)

    # ----segmenting gamma wavelet at theta phases ----------
    theta_params = sess.theta.getParams(rem_theta)
    bin_angle = np.linspace(0, 360, int(360 / 9) + 1)
    bin_ind = np.digitize(theta_params.angle, bin_angle)

    gamma_at_theta = pd.DataFrame()
    for i in np.unique(bin_ind):
        find_where = np.where(bin_ind == i)[0]
        gamma_at_theta[bin_angle[i - 1]] = np.mean(wav[:, find_where], axis=1)
    gamma_at_theta.insert(0, column="freq", value=frgamma)

    # ---- appending for all subjects ------------
    gamma_at_theta_all = gamma_at_theta_all.append(gamma_at_theta)

ax = fig.add_subplot(inner[:, :])
mean_gamma = (
    gamma_at_theta_all.groupby(by="freq").mean().transform(stats.zscore, axis=1)
)

sns.heatmap(mean_gamma, robust=True, cmap="Spectral_r", shading="gouraud", ax=ax)
ax.invert_yaxis()
ax.set_xlabel(r"$\theta$ phase")
ax.set_ylabel("Frequency (Hz)")


# endregion


# %% Detects slow gamma periods during MAZE and averages spectrogram around theta cycles
# region
cmap = mpl.cm.get_cmap("hot_r")
for sub, sess in enumerate(sessions):

    sess.trange = np.array([])
    eegSrate = sess.recinfo.lfpSrate
    posx = sess.position.x
    posy = sess.position.y
    post = sess.position.t
    maze = sess.epochs.maze

    lfp, _, _ = sess.ripple.best_chan_lfp()
    lfp = lfp[0, :]
    t = np.linspace(0, len(lfp) / 1250, len(lfp))

    tstart = maze[0]
    tend = maze[1]

    lfpmaze = lfp[(t > tstart) & (t < tend)]
    tmaze = np.linspace(tstart, tend, len(lfpmaze))
    posmazex = posx[(post > tstart) & (post < tend)]
    posmazey = posy[(post > tstart) & (post < tend)]
    postmaze = np.linspace(tstart, tend, len(posmazex))
    speed = np.sqrt(np.diff(posmazex) ** 2 + np.diff(posmazey) ** 2)
    speed = gaussian_filter1d(speed, sigma=10)

    frtheta = np.arange(5, 12, 0.5)
    wavdec = signal_process.wavelet_decomp(lfpmaze, freqs=frtheta)
    wav = wavdec.cohen(ncycles=7)
    # wavtheta = doWavelet(lfpmaze, freqs=frtheta, ncycles=3)

    sum_theta = gaussian_filter1d(np.sum(wav, axis=0), sigma=10)
    zsc_theta = stats.zscore(sum_theta)
    thetaevents = threshPeriods(
        zsc_theta, lowthresh=0, highthresh=1, minDistance=625, minDuration=2 * 1250
    )

    strong_theta = []
    for (beg, end) in thetaevents:
        strong_theta.extend(lfpmaze[beg:end])
    strong_theta = np.asarray(strong_theta)

    frgamma = np.arange(25, 150, 1)
    wavdec = signal_process.wavelet_decomp(strong_theta, freqs=frgamma)
    wav = wavdec.colgin2009()
    wav = stats.zscore(wav, axis=1)
    # wav = wavdec.colgin2009(ncycles=3)

    theta_filter = stats.zscore(
        signal_process.filter_sig.filter_cust(strong_theta, lf=5, hf=11)
    )
    hil_theta = signal_process.hilbertfast(theta_filter)
    theta_amp = np.abs(hil_theta)
    theta_angle = np.angle(hil_theta, deg=True) + 180

    theta_troughs = sg.find_peaks(-theta_filter)[0]

    avg_theta = np.zeros(156)
    mean_gamma = np.zeros((wav.shape[0], 156))
    for i in theta_troughs[1:]:
        mean_gamma = mean_gamma + wav[:, i - 125 : i + 31]
        avg_theta = avg_theta + strong_theta[i - 125 : i + 31]

    mean_gamma = mean_gamma / len(theta_troughs)

    plt.clf()
    fig = plt.figure(1, figsize=(10, 15))
    gs = gridspec.GridSpec(3, 1, figure=fig)
    fig.subplots_adjust(hspace=0.2)

    ax = fig.add_subplot(gs[0])
    t_thetacycle = np.linspace(-100, 25, 156)
    ax.pcolorfast(t_thetacycle, frgamma, mean_gamma)
    ax.set_xlim([-100, 25])
    ax.set_ylabel("Frequency (Hz)")

    ax = fig.add_subplot(gs[1], sharex=ax)
    ax.plot(t_thetacycle, avg_theta / len(theta_troughs))
    ax.set_xlabel("Time from theta trough (ms)")
    ax.set_ylabel("Amplitude")

    ax = fig.add_subplot(gs[2])
    todB = lambda power: 10 * np.log10(power)
    pxx_theta, f_theta = getPxx(strong_theta)
    ax.plot(f_theta, todB(pxx_theta), color="#ef253c", alpha=0.8)
    ax.set_xscale("log")
    ax.set_xlim([4, 220])
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")

    # thetaevents = thetaevents/eegSrate + tstart

# endregion

# %% Detects theta periods during Sleep deprivation and spectrogram around theta cycle
# region
cmap = mpl.cm.get_cmap("hot_r")
for sub, sess in enumerate(sessions):

    sess.trange = np.array([])
    eegSrate = sess.recinfo.lfpSrate
    posx = sess.position.x
    posy = sess.position.y
    post = sess.position.t
    post = sess.epochs.post

    lfp, _, _ = sess.ripple.best_chan_lfp()
    lfp = lfp[0, :]
    t = np.linspace(0, len(lfp) / 1250, len(lfp))

    tstart = post[0]
    tend = post[0] + 5 * 3600

    lfpSD = lfp[(t > tstart) & (t < tend)]
    tSD = np.linspace(tstart, tend, len(lfpSD))

    frtheta = np.arange(5, 12, 0.5)
    wavdec = wavelet_decomp(lfpSD, freqs=frtheta)
    wav = wavdec.cohen()
    # frgamma = np.arange(25, 50, 1)
    # wavdec = wavelet_decomp(lfpSD, freqs=frgamma)
    # wav = wavdec.colgin2009()
    # wavtheta = doWavelet(lfpSD, freqs=frtheta, ncycles=3)

    sum_theta = gaussian_filter1d(np.sum(wav, axis=0), sigma=10)
    zsc_theta = stats.zscore(sum_theta)
    thetaevents = threshPeriods(
        zsc_theta, lowthresh=0, highthresh=1.5, minDistance=300, minDuration=1250
    )

    strong_theta = []
    theta_indices = []
    for (beg, end) in thetaevents:
        strong_theta.extend(lfpSD[beg:end])
        theta_indices.extend(np.arange(beg, end))
    strong_theta = np.asarray(strong_theta)
    theta_indices = np.asarray(theta_indices)

    non_theta = np.delete(lfpSD, theta_indices)
    frgamma = np.arange(150, 250, 1)
    # frgamma = np.linspace(25, 150, 1)
    wavdec = wavelet_decomp(strong_theta, freqs=frgamma)
    wav = wavdec.colgin2009()
    # wav = wavdec.cohen(ncycles=7)
    wav = stats.zscore(wav)

    theta_filter = stats.zscore(filter_sig.filter_cust(strong_theta, lf=4, hf=11))
    hil_theta = hilbertfast(theta_filter)
    theta_amp = np.abs(hil_theta)
    theta_angle = np.angle(hil_theta, deg=True) + 180

    theta_troughs = sg.find_peaks(-theta_filter)[0]

    avg_theta = np.zeros(156)
    mean_gamma = np.zeros((wav.shape[0], 156))
    for i in theta_troughs[1:]:
        mean_gamma = mean_gamma + wav[:, i - 125 : i + 31]
        avg_theta = avg_theta + strong_theta[i - 125 : i + 31]

    mean_gamma = mean_gamma / len(theta_troughs)

    mean_gamma = stats.zscore(mean_gamma, axis=1)

    plt.clf()
    fig = plt.figure(1, figsize=(10, 15))
    gs = gridspec.GridSpec(3, 1, figure=fig)
    fig.subplots_adjust(hspace=0.2)

    ax = fig.add_subplot(gs[0])
    t_thetacycle = np.linspace(-100, 25, 156)
    ax.pcolorfast(t_thetacycle, frgamma, mean_gamma, cmap="jet")
    ax.set_ylabel("Frequency (Hz)")

    # ax.contourf(t_thetacycle,frgamma,mean_gamma)
    ax.set_xlim([-100, 25])
    ax = fig.add_subplot(gs[1], sharex=ax)
    ax.plot(t_thetacycle, avg_theta / len(theta_troughs), "k")
    ax.set_xlabel("Time from theta trough (ms)")
    ax.set_ylabel("Amplitude")

    ax = fig.add_subplot(gs[2])
    todB = lambda power: 10 * np.log10(power)
    pxx_theta, f_theta = getPxx(strong_theta)
    pxx_nontheta, f_nontheta = getPxx(non_theta)
    ax.plot(f_theta, todB(pxx_theta), color="#ef253c", alpha=0.8)
    ax.plot(f_theta, todB(pxx_nontheta), color="#4e5c73", alpha=0.8)
    ax.set_xscale("log")
    ax.set_xlim([4, 220])
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    ax.legend(["Theta", "non-theta"])

    # thetaevents = thetaevents/eegSrate + tstart
# endregion

#%% Spectrogram around theta cycles using fourier based analyses
# region
cmap = mpl.cm.get_cmap("hot_r")
for sub, sess in enumerate(sessions):

    sess.trange = np.array([])
    eegSrate = sess.recinfo.lfpSrate
    posx = sess.position.x
    posy = sess.position.y
    post = sess.position.t
    maze = sess.epochs.maze

    lfp, _, _ = sess.ripple.best_chan_lfp()
    lfp = lfp[0, :]
    t = np.linspace(0, len(lfp) / 1250, len(lfp))

    tstart = maze[0]
    tend = maze[1]

    lfpSD = lfp[(t > tstart) & (t < tend)]
    tmaze = np.linspace(tstart, tend, len(lfpSD))
    posmazex = posx[(post > tstart) & (post < tend)]
    posmazey = posy[(post > tstart) & (post < tend)]
    postmaze = np.linspace(tstart, tend, len(posmazex))
    speed = np.sqrt(np.diff(posmazex) ** 2 + np.diff(posmazey) ** 2)
    speed = gaussian_filter1d(speed, sigma=10)

    frtheta = np.arange(5, 12, 0.5)
    wavdec = signal_process.wavelet_decomp(lfpSD, freqs=frtheta)
    wav = wavdec.cohen()
    # frgamma = np.arange(25, 50, 1)
    # wavdec = wavelet_decomp(lfpSD, freqs=frgamma)
    # wav = wavdec.colgin2009()
    # wavtheta = doWavelet(lfpSD, freqs=frtheta, ncycles=3)

    sum_theta = gaussian_filter1d(np.sum(wav, axis=0), sigma=10)
    zsc_theta = stats.zscore(sum_theta)
    thetaevents = threshPeriods(
        zsc_theta, lowthresh=0, highthresh=1.5, minDistance=300, minDuration=1250
    )

    strong_theta = []
    theta_indices = []
    for (beg, end) in thetaevents:
        strong_theta.extend(lfpSD[beg:end])
        theta_indices.extend(np.arange(beg, end))
    strong_theta = np.asarray(strong_theta)
    theta_indices = np.asarray(theta_indices)

    non_theta = np.delete(lfpSD, theta_indices)
    frgamma = np.arange(25, 150, 1)
    specgram = signal_process.spectrogramBands(strong_theta)
    fr_sxx = specgram.freq
    fr_gamma_indx = np.where((fr_sxx > 25) & (fr_sxx < 150))
    wav = specgram.sxx[fr_gamma_indx, :]
    wav = stats.zscore(wav)

    theta_filter = stats.zscore(
        signal_process.filter_sig.filter_cust(strong_theta, lf=4, hf=11)
    )
    hil_theta = signal_process.hilbertfast(theta_filter)
    theta_amp = np.abs(hil_theta)
    theta_angle = np.angle(hil_theta, deg=True) + 180

    theta_troughs = sg.find_peaks(-theta_filter)[0]

    avg_theta = np.zeros(156)
    mean_gamma = np.zeros((wav.shape[0], 156))
    for i in theta_troughs[1:]:
        mean_gamma = mean_gamma + wav[:, i - 125 : i + 31]
        avg_theta = avg_theta + strong_theta[i - 125 : i + 31]

    mean_gamma = mean_gamma / len(theta_troughs)

    mean_gamma = stats.zscore(mean_gamma, axis=1)

    plt.clf()
    fig = plt.figure(1, figsize=(10, 15))
    gs = gridspec.GridSpec(3, 1, figure=fig)
    fig.subplots_adjust(hspace=0.2)

    ax = fig.add_subplot(gs[0])
    t_thetacycle = np.linspace(-100, 25, 156)
    ax.pcolorfast(t_thetacycle, frgamma, mean_gamma, cmap="jet")
    ax.set_ylabel("Frequency (Hz)")

    # ax.contourf(t_thetacycle,frgamma,mean_gamma)
    ax.set_xlim([-100, 25])
    ax = fig.add_subplot(gs[1], sharex=ax)
    ax.plot(t_thetacycle, avg_theta / len(theta_troughs), "k")
    ax.set_xlabel("Time from theta trough (ms)")
    ax.set_ylabel("Amplitude")

    ax = fig.add_subplot(gs[2])
    todB = lambda power: 10 * np.log10(power)
    pxx_theta, f_theta = getPxx(strong_theta)
    pxx_nontheta, f_nontheta = getPxx(non_theta)
    ax.plot(f_theta, todB(pxx_theta), color="#ef253c", alpha=0.8)
    ax.plot(f_theta, todB(pxx_nontheta), color="#4e5c73", alpha=0.8)
    ax.set_xscale("log")
    ax.set_xlim([4, 220])
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    ax.legend(["Theta", "non-theta"])

    # thetaevents = thetaevents/eegSrate + tstart
# endregion

#%% bicoherence analysis during strong theta periods
# region
plt.clf()
fig = plt.figure(1, figsize=(10, 15))
gs = gridspec.GridSpec(2, 2, figure=fig)
fig.subplots_adjust(hspace=0.3)
cmap = mpl.cm.get_cmap("hot_r")
for sub, sess in enumerate(sessions):

    sess.trange = np.array([])
    eegSrate = sess.recinfo.lfpSrate
    # posx = sess.position.x
    # posy = sess.position.y
    # post = sess.position.t
    maze = sess.epochs.maze

    lfp, _, _ = sess.ripple.best_chan_lfp()
    lfp = lfp[0, :]
    t = np.linspace(0, len(lfp) / 1250, len(lfp))

    tstart = maze[0]
    tend = maze[1]

    lfpmaze = lfp[(t > tstart) & (t < tend)]
    tmaze = np.linspace(tstart, tend, len(lfpmaze))
    # posmazex = posx[(post > tstart) & (post < tend)]
    # posmazey = posy[(post > tstart) & (post < tend)]
    # postmaze = np.linspace(tstart, tend, len(posmazex))
    # speed = np.sqrt(np.diff(posmazex) ** 2 + np.diff(posmazey) ** 2)
    # speed = gaussian_filter1d(speed, sigma=10)

    frtheta = np.arange(5, 12, 0.5)
    wavdec = signal_process.wavelet_decomp(lfpmaze, freqs=frtheta)
    wav = wavdec.cohen()
    # frgamma = np.arange(25, 50, 1)
    # wavdec = wavelet_decomp(lfpmaze, freqs=frgamma)
    # wav = wavdec.colgin2009()
    # wavtheta = doWavelet(lfpmaze, freqs=frtheta, ncycles=3)

    sum_theta = gaussian_filter1d(np.sum(wav, axis=0), sigma=10)
    zsc_theta = stats.zscore(sum_theta)
    thetaevents = threshPeriods(
        zsc_theta, lowthresh=0, highthresh=0.5, minDistance=300, minDuration=1250
    )

    strong_theta = []
    theta_indices = []
    for (beg, end) in thetaevents:
        strong_theta.extend(lfpmaze[beg:end])
        theta_indices.extend(np.arange(beg, end))
    strong_theta = np.asarray(strong_theta)
    theta_indices = np.asarray(theta_indices)

    non_theta = np.delete(lfpmaze, theta_indices)

    strong_theta = strong_theta - np.mean(strong_theta)
    strong_theta = sg.detrend(strong_theta, type="linear")
    # strong_theta = stats.zscore(strong_theta)
    bicoh, bicoh_freq = signal_process.bicoherence(
        strong_theta, window=4 * 1250, overlap=2 * 1250
    )

    ax = fig.add_subplot(gs[sub])
    ax.pcolorfast(bicoh_freq, bicoh_freq, bicoh, cmap="YlGn", vmax=0.2)
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Frequency (Hz)")
    # plt.pcolormesh(bispec_freq, bispec_freq, bispec, vmin=0, vmax=0.1, cmap="YlGn")
    ax.set_ylim([2, 75])

    ax = fig.add_subplot(gs[sub + 2])
    f, t, sxx = sg.spectrogram(strong_theta, nperseg=1250, noverlap=625, fs=1250)
    ax.pcolorfast(t, f, sxx, cmap="YlGn", vmax=0.05)
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (s)")
    # plt.pcolormesh(bispec_freq, bispec_freq, bispec, vmin=0, vmax=0.1, cmap="YlGn")
    ax.set_ylim([1, 75])

fig.suptitle("fourier and bicoherence analysis of strong theta during MAZE")

# plt.clf()
# fig = plt.figure(1, figsize=(10, 15))
# gs = gridspec.GridSpec(3, 1, figure=fig)
# fig.subplots_adjust(hspace=0.2)

# # ax.contourf(t_thetacycle,frgamma,mean_gamma)
# ax.set_xlim([-100, 25])
# ax = fig.add_subplot(gs[1], sharex=ax)
# ax.plot(t_thetacycle, avg_theta / len(theta_troughs), "k")
# ax.set_xlabel("Time from theta trough (ms)")
# ax.set_ylabel("Amplitude")

# ax = fig.add_subplot(gs[2])
# todB = lambda power: 10 * np.log10(power)
# pxx_theta, f_theta = getPxx(strong_theta)
# pxx_nontheta, f_nontheta = getPxx(non_theta)
# ax.plot(f_theta, todB(pxx_theta), color="#ef253c", alpha=0.8)
# ax.plot(f_theta, todB(pxx_nontheta), color="#4e5c73", alpha=0.8)
# ax.set_xscale("log")
# ax.set_xlim([4, 220])
# ax.set_xlabel("Frequency (Hz)")
# ax.set_ylabel("Power")
# ax.legend(["Theta", "non-theta"])

# thetaevents = thetaevents/eegSrate + tstart
# endregion


#%% during Sleep deprivation compare theta phase-gamma amplitude relationship
# region
group = []
plt.clf()
fig = plt.figure(1, figsize=(1, 15))
gs = gridspec.GridSpec(3, 1, figure=fig)
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

        theta_lfp = stats.zscore(signal_process.filter_sig.filter_theta(lfp_bin))
        gamma_lfp = stats.zscore(signal_process.filter_sig.filter_gamma(lfp_bin))

        hil_theta = signal_process.hilbertfast(theta_lfp)
        hil_gamma = signal_process.hilbertfast(gamma_lfp)

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
# endregion

#%% Compare theta-gamma-phase coupling during REM
# region
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

    lfprem = []
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
        ax.set_xlabel("Degrees (from theta trough)")
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
# endregion
