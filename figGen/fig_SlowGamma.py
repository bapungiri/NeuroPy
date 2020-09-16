#%%
import numpy as np
from callfunc import processData
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.stats as stats
import pandas as pd
import seaborn as sns
import signal_process
import matplotlib as mpl
import scipy.signal as sg
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from mathutil import threshPeriods

basePath = [
    "/data/Clustering/SleepDeprivation/RatJ/Day1/",
    "/data/Clustering/SleepDeprivation/RatK/Day1/",
    "/data/Clustering/SleepDeprivation/RatN/Day1/",
    "/data/Clustering/SleepDeprivation/RatJ/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day2/",
    "/data/Clustering/SleepDeprivation/RatN/Day2/",
]


sessions = [processData(_) for _ in basePath]

plt.clf()
fig = plt.figure(1, figsize=(10, 15))
gs = gridspec.GridSpec(3, 3, figure=fig)
fig.subplots_adjust(hspace=0.3, wspace=0.3)
fig.suptitle("Slow gamma related")
titlesize = 8


# # %% Detects strong theta within maze and averages spectrogram around theta cycles using Wavelets with bicoherence plot
# region
# inner = gridspec.GridSpecFromSubplotSpec(
#     2, 1, subplot_spec=gs[0, 0], wspace=0.1, hspace=0.1
# )
# cmap = mpl.cm.get_cmap("hot_r")
# strong_theta = []
# for sub, sess in enumerate(sessions):

#     sess.trange = np.array([])
#     eegSrate = sess.recinfo.lfpSrate
#     posx = sess.position.x
#     posy = sess.position.y
#     post = sess.position.t
#     maze = sess.epochs.maze

#     lfp, _, _ = sess.ripple.best_chan_lfp()
#     lfp = lfp[0, :]
#     t = np.linspace(0, len(lfp) / 1250, len(lfp))

#     tstart = maze[0]
#     tend = maze[1]

#     lfpmaze = lfp[(t > tstart) & (t < tend)]
#     tmaze = np.linspace(tstart, tend, len(lfpmaze))
#     posmazex = posx[(post > tstart) & (post < tend)]
#     posmazey = posy[(post > tstart) & (post < tend)]
#     postmaze = np.linspace(tstart, tend, len(posmazex))
#     speed = np.sqrt(np.diff(posmazex) ** 2 + np.diff(posmazey) ** 2)
#     speed = gaussian_filter1d(speed, sigma=10)

#     frtheta = np.arange(5, 12, 0.5)
#     wavdec = signal_process.wavelet_decomp(lfpmaze, freqs=frtheta)
#     wav = wavdec.cohen()
#     # frgamma = np.arange(25, 50, 1)
#     # wavdec = wavelet_decomp(lfpmaze, freqs=frgamma)
#     # wav = wavdec.colgin2009()
#     # wavtheta = doWavelet(lfpmaze, freqs=frtheta, ncycles=3)

#     sum_theta = gaussian_filter1d(np.sum(wav, axis=0), sigma=10)
#     zsc_theta = stats.zscore(sum_theta)
#     thetaevents = threshPeriods(
#         zsc_theta, lowthresh=0, highthresh=1.5, minDistance=300, minDuration=1250
#     )

#     # strong_theta = []
#     theta_indices = []
#     for (beg, end) in thetaevents:
#         strong_theta.extend(lfpmaze[beg:end])
#         theta_indices.extend(np.arange(beg, end))
#     strong_theta.extend(np.asarray(strong_theta))
#     # theta_indices = np.asarray(theta_indices)
#     # non_theta = np.delete(lfpmaze, theta_indices)


# frgamma = np.arange(25, 150, 1)
# # frgamma = np.linspace(25, 150, 1)
# strong_theta = np.asarray(strong_theta)
# wavdec = signal_process.wavelet_decomp(strong_theta, freqs=frgamma)
# wav = wavdec.colgin2009()
# # wav = wavdec.cohen(ncycles=7)

# wav = stats.zscore(wav, axis=1)

# theta_filter = stats.zscore(
#     signal_process.filter_sig.filter_cust(strong_theta, lf=4, hf=11)
# )

# hil_theta = signal_process.hilbertfast(theta_filter)
# theta_amp = np.abs(hil_theta)
# theta_angle = np.angle(hil_theta, deg=True) + 180
# theta_troughs = sg.find_peaks(-theta_filter)[0]

# bin_angle = np.linspace(0, 360, int(360 / 9) + 1)
# bin_ind = np.digitize(theta_angle, bin_angle)

# wav_phase = []
# for i in np.unique(bin_ind):
#     find_where = np.where(bin_ind == i)[0]
#     wav_at_angle = np.mean(wav[:, find_where], axis=1)
#     wav_phase.append(wav_at_angle)

# wav_phase = np.asarray(wav_phase).T

# ax = fig.add_subplot(inner[:, :])
# ax.pcolorfast(bin_angle[:-1], frgamma[:-1], wav_phase, cmap="Spectral_r")
# ax.set_xlabel(r"$\theta$ phase")

# ax.set_ylabel("frequency (Hz)")

# bicoh, freq, bispec = signal_process.bicoherence(strong_theta, fhigh=100)

# # bicoh = gaussian_filter(bicoh, sigma=2)
# # bicoh = np.where(bicoh > 0.05, bicoh, 0)
# bispec_real = gaussian_filter(np.real(bispec), sigma=2)
# bispec_imag = gaussian_filter(np.imag(bispec), sigma=2)
# bispec_angle = gaussian_filter(np.angle(bispec, deg=True), sigma=2)

# ax = fig.add_subplot(gs[0, 1])
# ax.clear()
# im = ax.pcolorfast(freq, freq, bicoh, cmap="Spectral_r", vmax=0.05, vmin=0)
# # ax.contour(freq, freq, bicoh, levels=[0.1, 0.2, 0.3], colors="k", linewidths=1)
# ax.set_ylim([1, max(freq) / 2])
# ax.set_xlabel("Frequency (Hz)")
# ax.set_ylabel("Frequency (Hz)")


# # cax = fig.add_axes([0.3, 0.8, 0.5, 0.05])
# # cax.clear()
# # ax.contour(freq, freq, bicoh, levels=[0.1, 0.2, 0.3], colors="k", linewidths=1)
# fig.colorbar(im, ax=ax, orientation="horizontal")
# # avg_theta = np.zeros(156)
# # mean_gamma = np.zeros((wav.shape[0], 156))
# # for i in theta_troughs[1:]:
# #     mean_gamma = mean_gamma + wav[:, i - 125 : i + 31]
# #     avg_theta = avg_theta + strong_theta[i - 125 : i + 31]

# # mean_gamma = mean_gamma / len(theta_troughs)
# # mean_gamma = stats.zscore(mean_gamma, axis=1)


# # ax = fig.add_subplot(inner[0])
# # t_thetacycle = np.linspace(-100, 25, 156)
# # ax.pcolorfast(t_thetacycle, frgamma, mean_gamma, cmap="Spectral")
# # ax.set_ylabel("Frequency (Hz)")

# # # ax.contourf(t_thetacycle,frgamma,mean_gamma)
# # ax.set_xlim([-100, 25])
# # ax = fig.add_subplot(inner[1], sharex=ax)
# # ax.plot(t_thetacycle, avg_theta / len(theta_troughs), "k")
# # ax.set_xlabel("Time from theta trough (ms)")
# # ax.set_ylabel("Amplitude")

# endregion

#%% Average wavelet spectrogram around theta cycle during REM sleep
# region

# endregion


#%% theta phase specific extraction of lfp during strong theta MAZE
# region


all_theta = []
cmap = mpl.cm.get_cmap("Set3")
for sub, sess in enumerate(sessions[4:5]):

    sess.trange = np.array([])
    eegSrate = sess.recinfo.lfpSrate
    maze = sess.epochs.maze

    lfpmaze = sess.utils.geteeg(sess.theta.bestchan, timeRange=maze)
    lfpmaze_t = np.linspace(maze[0], maze[1], len(lfpmaze))

    thetalfp = signal_process.filter_sig.filter_cust(lfpmaze, lf=4, hf=10)
    hil_theta = signal_process.hilbertfast(thetalfp)
    theta_amp = np.abs(hil_theta)

    zsc_theta = stats.zscore(theta_amp)
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

    theta_lfp = stats.zscore(signal_process.filter_sig.filter_theta(strong_theta))
    gamma_lfp = stats.zscore(
        signal_process.filter_sig.filter_cust(strong_theta, lf=25, hf=50)
    )

    # filt_theta = signal_process.filter_sig.filter_cust(theta_lfp, lf=20, hf=60)
    hil_theta = signal_process.hilbertfast(theta_lfp)
    theta_amp = np.abs(hil_theta)
    theta_angle = np.angle(hil_theta, deg=True) + 180
    angle_bin = np.linspace(0, 360, 10)  # divide into 5 bins so each bin=25ms
    bin_ind = np.digitize(theta_angle, bins=angle_bin)

    ax = fig.add_subplot(gs[0])
    # ax_ = fig.add_subplot(gs[1, 0])
    axins = ax.inset_axes([0.5, 0.5, 0.47, 0.47])
    for phase in range(1, len(angle_bin)):
        strong_theta_atphase = gamma_lfp[np.where(bin_ind == phase)[0]]
        # strong_theta_atphase = signal_process.filter_sig.filter_cust(
        #     strong_theta_atphase, lf=20, hf=100
        # )

        # ax = fig.add_subplot(gs[phase - 1])
        f, t, sxx = sg.spectrogram(
            strong_theta_atphase, nperseg=1250, noverlap=625, fs=1250
        )
        f_, pxx = sg.welch(strong_theta_atphase, nperseg=1250, noverlap=625, fs=1250)
        # ax.pcolorfast(t, f, stats.zscore(sxx, axis=1), cmap="YlGn")

        ax.plot(
            f_,
            # np.mean(sxx, axis=1),
            pxx,
            color=cmap(phase),
            label=f"{int(angle_bin[phase-1])}-{int(angle_bin[phase])}",
        )
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Mean amplitude across time")
        # plt.pcolormesh(bispec_freq, bispec_freq, bispec, vmin=0, vmax=0.1, cmap="YlGn")
        ax.set_xlim([2, 120])

        axins.plot(
            [angle_bin[phase - 1], angle_bin[phase]], [1, 1], color=cmap(phase), lw=2
        )

        # ax_.plot(f_, pxx)

    axins.axis("off")
    # ax.legend(title="Theta Phase")
    ax.set_title("Mean power spectrum by breaking \n down theta signal by phase")


# fig.suptitle("fourier and bicoherence analysis of strong theta during MAZE")
# endregion


#%% theta phase specific extraction of lfp during strong theta MAZE
# region

all_theta = []
cmap = mpl.cm.get_cmap("Set3")
for sub, sess in enumerate(sessions[4:5]):

    sess.trange = np.array([])
    eegSrate = sess.recinfo.lfpSrate
    maze = sess.epochs.maze

    lfpmaze = sess.utils.geteeg(sess.theta.bestchan, timeRange=maze)
    lfpmaze_t = np.linspace(maze[0], maze[1], len(lfpmaze))

    thetalfp = signal_process.filter_sig.filter_cust(lfpmaze, lf=4, hf=10)
    hil_theta = signal_process.hilbertfast(thetalfp)
    theta_amp = np.abs(hil_theta)

    zsc_theta = stats.zscore(theta_amp)
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

    theta_lfp = stats.zscore(signal_process.filter_sig.filter_theta(strong_theta))
    gamma_lfp = stats.zscore(
        signal_process.filter_sig.filter_cust(strong_theta, lf=25, hf=50)
    )
    # filt_theta = signal_process.filter_sig.filter_cust(theta_lfp, lf=20, hf=60)
    hil_theta = signal_process.hilbertfast(theta_lfp)
    theta_amp = np.abs(hil_theta)
    theta_angle = np.angle(hil_theta, deg=True) + 180
    angle_bin = np.linspace(0, 360, 6)  # divide into 5 bins so each bin=25ms
    bin_ind = np.digitize(theta_angle, bins=angle_bin)

    ax = fig.add_subplot(gs[1])
    axins = ax.inset_axes([0.5, 0.5, 0.47, 0.47])
    for phase in range(1, len(angle_bin)):
        strong_theta_atphase = gamma_lfp[np.where(bin_ind == phase)[0]]
        # strong_theta_atphase = signal_process.filter_sig.filter_cust(
        #     strong_theta_atphase, lf=20, hf=100
        # )

        # ax = fig.add_subplot(gs[phase - 1])
        f, t, sxx = sg.spectrogram(
            strong_theta_atphase, nperseg=1250, noverlap=625, fs=1250
        )
        f_, pxx = sg.welch(strong_theta_atphase, nperseg=1250, noverlap=625, fs=1250)
        # ax.pcolorfast(t, f, stats.zscore(sxx, axis=1), cmap="YlGn")

        ax.plot(
            f_,
            # np.mean(sxx, axis=1),
            pxx,
            color=cmap(phase),
            label=f"{int(angle_bin[phase-1])}-{int(angle_bin[phase])}",
        )
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Mean amplitude across time")
        # plt.pcolormesh(bispec_freq, bispec_freq, bispec, vmin=0, vmax=0.1, cmap="YlGn")
        ax.set_xlim([2, 100])

        axins.plot(
            [angle_bin[phase - 1], angle_bin[phase]], [1, 1], color=cmap(phase), lw=2
        )

    axins.axis("off")
    # ax.legend(title="Theta Phase")
    ax.set_title("Mean power spectrum by breaking \n down theta signal by phase")


# fig.suptitle("fourier and bicoherence analysis of strong theta during MAZE")
# endregion


#%% theta phase specific extraction of lfp during strong theta MAZE
# region

all_theta = []
cmap = mpl.cm.get_cmap("jet")
for sub, sess in enumerate(sessions[4:5]):

    sess.trange = np.array([])
    eegSrate = sess.recinfo.lfpSrate
    maze = sess.epochs.maze

    lfpmaze = sess.utils.geteeg(sess.theta.bestchan, timeRange=maze)
    lfpmaze_t = np.linspace(maze[0], maze[1], len(lfpmaze))

    thetalfp = signal_process.filter_sig.filter_cust(lfpmaze, lf=4, hf=10)
    hil_theta = signal_process.hilbertfast(thetalfp)
    theta_amp = np.abs(hil_theta)

    zsc_theta = stats.zscore(theta_amp)
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

    theta_lfp = stats.zscore(signal_process.filter_sig.filter_theta(strong_theta))
    gamma_lfp = stats.zscore(
        signal_process.filter_sig.filter_cust(strong_theta, lf=25, hf=50)
    )
    # filt_theta = signal_process.filter_sig.filter_cust(theta_lfp, lf=20, hf=60)
    hil_theta = signal_process.hilbertfast(theta_lfp)
    theta_amp = np.abs(hil_theta)
    theta_angle = np.angle(hil_theta, deg=True) + 180
    angle_bin = np.arange(0, 360 - 40, 5)  # divide into 5 bins so each bin=25ms
    bin_ind = np.digitize(theta_angle, bins=angle_bin)

    ax = fig.add_subplot(gs[2])
    # axins = ax.inset_axes([0.5, 0.5, 0.47, 0.47])
    for phase in angle_bin:
        strong_theta_atphase = gamma_lfp[
            np.where((theta_angle > phase) & (theta_angle < phase + 40))[0]
        ]
        # strong_theta_atphase = signal_process.filter_sig.filter_cust(
        #     strong_theta_atphase, lf=20, hf=100
        # )

        # ax = fig.add_subplot(gs[phase - 1])
        f, t, sxx = sg.spectrogram(
            strong_theta_atphase, nperseg=1250, noverlap=625, fs=1250
        )

        f_, pxx = sg.welch(strong_theta_atphase, nperseg=1250, noverlap=625, fs=1250)
        # ax.pcolorfast(t, f, stats.zscore(sxx, axis=1), cmap="YlGn")

        ax.plot(
            f_,
            pxx,
            color=cmap(phase),
            # label=f"{int(angle_bin[phase-1])}-{int(angle_bin[phase])}",
        )
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Mean amplitude across time")
        # plt.pcolormesh(bispec_freq, bispec_freq, bispec, vmin=0, vmax=0.1, cmap="YlGn")
        ax.set_xlim([2, 100])

        # axins.plot(
        #     [angle_bin[phase - 1], angle_bin[phase]], [1, 1], color=cmap(phase), lw=2
        # )

    # axins.axis("off")
    # ax.legend(title="Theta Phase")
    ax.set_title("Mean power spectrum by breaking \n down theta signal by phase")


# fig.suptitle("fourier and bicoherence analysis of strong theta during MAZE")
# endregion


#%% Multiple regression analysis on slow gamma power explained by variables such as theta-harmonic, theta-asymmetry, speed etc. Also comparing it with theta-harmonic being explained by similar variables
# region
# plt.clf()
# fig, ax = plt.subplots(1, 2, num=1, sharey=True)
exp_var_gamma_all, exp_var_harmonic_all = [], []
for sub, sess in enumerate(sessions):
    sess.trange = np.array([])
    maze = sess.epochs.maze
    speed = sess.position.speed
    t_position = sess.position.t[1:]
    deadtime = sess.artifact.time

    lfpmaze = sess.utils.geteeg(sess.theta.bestchan, timeRange=maze)
    lfpmaze_t = np.linspace(maze[0], maze[1], len(lfpmaze))
    speed = np.interp(lfpmaze_t, t_position, speed)

    if deadtime is not None:
        dead_indx = np.concatenate(
            [
                np.where((lfpmaze_t > start) & (lfpmaze_t < end))[0]
                for (start, end) in deadtime
            ]
        )
        lfpmaze = np.delete(lfpmaze, dead_indx)
        speed = np.delete(speed, dead_indx)

    # --- calculating theta parameters ---------
    thetalfp = signal_process.filter_sig.filter_cust(lfpmaze, lf=4, hf=10)
    hil_theta = signal_process.hilbertfast(thetalfp)
    theta_angle = np.abs(np.angle(hil_theta, deg=True))
    theta_trough = sg.find_peaks(theta_angle)[0]
    theta_peak = sg.find_peaks(-theta_angle)[0]
    theta_amp = np.abs(hil_theta) ** 2

    # --- calculating slow gamma parameters -------
    gammalfp = signal_process.filter_sig.filter_cust(lfpmaze, lf=25, hf=50)
    hil_gamma = signal_process.hilbertfast(gammalfp)
    gamma_amp = np.abs(hil_gamma) ** 2

    # --- theta harmonic ----------
    theta_harmonic = signal_process.filter_sig.filter_cust(lfpmaze, lf=10, hf=22)
    hil_theta_harmonic = signal_process.hilbertfast(theta_harmonic)
    theta_harmonic_amp = np.abs(hil_theta_harmonic) ** 2

    if theta_peak[0] < theta_trough[0]:
        theta_peak = theta_peak[1:]
    if theta_trough[-1] > theta_peak[-1]:
        theta_trough = theta_trough[:-1]

    assert len(theta_trough) == len(theta_peak)

    rising_time = (theta_peak[1:] - theta_trough[1:]) / 1250
    falling_time = (theta_trough[1:] - theta_peak[:-1]) / 1250
    rise_fall = rising_time / (rising_time + falling_time)

    rise_midpoints = np.array(
        [
            trough
            + np.argmin(
                np.abs(
                    thetalfp[trough:peak]
                    - (max(thetalfp[trough:peak]) - np.ptp(thetalfp[trough:peak]) / 2)
                )
            )
            for (trough, peak) in zip(theta_trough, theta_peak)
        ]
    )

    fall_midpoints = np.array(
        [
            peak
            + np.argmin(
                np.abs(
                    thetalfp[peak:trough]
                    - (max(thetalfp[peak:trough]) - np.ptp(thetalfp[peak:trough]) / 2)
                )
            )
            for (peak, trough) in zip(theta_peak[:-1], theta_trough[1:])
        ]
    )
    peak_width = fall_midpoints - rise_midpoints[:-1]
    trough_width = rise_midpoints[1:] - fall_midpoints
    peak_trough_asymm = peak_width / (peak_width + trough_width)

    speed_in_theta = stats.binned_statistic(
        np.arange(len(thetalfp)), speed, bins=theta_trough
    )[0]
    thetapower_in_theta = stats.binned_statistic(
        np.arange(len(thetalfp)), theta_amp, bins=theta_trough
    )[0]
    gammapower_in_theta = stats.binned_statistic(
        np.arange(len(thetalfp)), gamma_amp, bins=theta_trough
    )[0]
    thetaharmonicpower_in_theta = stats.binned_statistic(
        np.arange(len(thetalfp)), theta_harmonic_amp, bins=theta_trough
    )[0]

    data = pd.DataFrame(
        {
            "gammaPower": gammapower_in_theta,
            "thetaharmonicPower": thetaharmonicpower_in_theta,
            "thetaPower": thetapower_in_theta,
            "speed": speed_in_theta,
            "asymm": rise_fall,
            "peaktrough": peak_trough_asymm,
        }
    )

    variables1 = data.columns.tolist()[1:]
    par_corr_stats_gamma = [
        data.partial_corr(
            y="gammaPower", x=var, covar=list(set(variables1) - set([var]))
        )
        for var in variables1
    ]

    variables2 = data.columns.tolist()[2:]
    par_corr_stats_harmonic = [
        data.partial_corr(
            y="thetaharmonicPower", x=var, covar=list(set(variables2) - set([var]))
        )
        for var in variables2
    ]

    exp_var_gamma = np.array([stat_.r2[0] * 100 for stat_ in par_corr_stats_gamma])
    p_val_gamma = np.array([stat_["p-val"][0] for stat_ in par_corr_stats_gamma])

    exp_var_harmonic = np.array(
        [stat_.r2[0] * 100 for stat_ in par_corr_stats_harmonic]
    )
    p_val_harmonic = np.array([stat_["p-val"][0] for stat_ in par_corr_stats_harmonic])

    exp_var_gamma_all.append(
        pd.DataFrame(exp_var_gamma[np.newaxis, :], columns=variables1)
    )
    exp_var_harmonic_all.append(
        pd.DataFrame(exp_var_harmonic[np.newaxis, :], columns=variables2)
    )


exp_var_gamma_all = pd.concat(exp_var_gamma_all)
# sns.barplot(ax=ax[0], data=exp_var_gamma_all, ci=None)
ax1 = plt.subplot(gs[1, 0])
ax1.bar(
    exp_var_gamma_all.columns.tolist(),
    exp_var_gamma_all.mean().values,
    # fmt="None",
    yerr=exp_var_gamma_all.sem().values,
    ecolor="black",
    capsize=10,
    edgecolor="k",
    color="#ffa69e",
)
ax1.tick_params(axis="x", labelrotation=90)
ax1.set_ylabel("Explained variance (%)")
ax1.set_title("Slow-gamma")

exp_var_harmonic_all = pd.concat(exp_var_harmonic_all)
# sns.barplot(ax=ax[1], data=exp_var_harmonic_all, ci=None)
ax2 = plt.subplot(gs[1, 1], sharey=ax1)
ax2.bar(
    exp_var_harmonic_all.columns.tolist(),
    exp_var_harmonic_all.mean().values,
    yerr=exp_var_harmonic_all.sem().values,
    ecolor="black",
    capsize=10,
    edgecolor="k",
    color="#ffbf00",
)
ax2.tick_params(axis="x", labelrotation=90)
ax2.set_title("Theta harmonic (10-22 Hz)")


# endregion

