#%%
import os

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import scipy.signal as sg
import scipy.stats as stats
import seaborn as sns
from scipy.ndimage import gaussian_filter, gaussian_filter1d

import signal_process
from callfunc import processData
from mathutil import threshPeriods
from plotUtil import savefig

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
fig = plt.figure(1, figsize=(8.5, 11))
fig.set_size_inches(8.5, 11)
gs = gridspec.GridSpec(3, 3, figure=fig)
fig.subplots_adjust(hspace=0.3, wspace=0.3)
fig.suptitle("Slow gamma related")
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


#%% Detects strong theta within maze and averages spectrogram around theta cycles using Wavelets
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

#%% Average wavelet spectrogram around theta cycle during REM sleep
# region

# endregion


#%% theta phase specific extraction of lfp during strong theta MAZE with different binning techinques
# region


axbin1 = plt.subplot(gs[0])
axbin1.clear()
axbin2 = plt.subplot(gs[1])
axbin2.clear()
axslide = plt.subplot(gs[2])
axslide.clear()

all_theta = []
bin1Data = pd.DataFrame()
bin2Data = pd.DataFrame()
slideData = pd.DataFrame()
cmap = mpl.cm.get_cmap("Set3")
for sub, sess in enumerate(sessions[4:5]):

    sess.trange = np.array([])
    eegSrate = sess.recinfo.lfpSrate
    maze = sess.epochs.maze

    lfpmaze = sess.utils.geteeg(sess.theta.bestchan, timeRange=maze)
    lfpmaze_t = np.linspace(maze[0], maze[1], len(lfpmaze))

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

    # ---- filtering strong theta periods into theta and gamma band ------
    theta_lfp = stats.zscore(
        signal_process.filter_sig.bandpass(strong_theta, lf=4, hf=10)
    )
    gamma_lfp = stats.zscore(
        signal_process.filter_sig.highpass(strong_theta, cutoff=25)
    )

    # ----- phase detection for theta band -----------
    # filt_theta = signal_process.filter_sig.filter_cust(theta_lfp, lf=20, hf=60)
    hil_theta = signal_process.hilbertfast(theta_lfp)
    theta_amp = np.abs(hil_theta)
    theta_angle = np.angle(hil_theta, deg=True) + 180  # range from 0 to 360

    """
    phase specific extraction of highpass filtered strong theta periods (>25 Hz) and concatenating similar phases across multiple theta cycles
    """

    # ----- dividing 360 degress into non-overlapping 5 bins ------------
    angle_bin = np.linspace(0, 360, 6)  # 5 bins so each bin=25ms
    angle_centers = angle_bin + np.diff(angle_bin).mean() / 2
    bin_ind = np.digitize(theta_angle, bins=angle_bin)
    df1 = pd.DataFrame()
    for phase in range(1, len(angle_bin)):
        strong_theta_atphase = gamma_lfp[np.where(bin_ind == phase)[0]]
        f_, pxx = sg.welch(strong_theta_atphase, nperseg=1250, noverlap=625, fs=1250)
        df1["freq"] = f_
        df1[str(angle_centers[phase - 1])] = pxx
    bin1Data = bin1Data.append(df1)

    # ----- dividing 360 degress into non-overlapping 9 bins ------------
    angle_bin = np.linspace(0, 360, 10)  # 9 bins
    angle_centers = angle_bin + np.diff(angle_bin).mean() / 2
    bin_ind = np.digitize(theta_angle, bins=angle_bin)
    df2 = pd.DataFrame()
    for phase in range(1, len(angle_bin)):
        strong_theta_atphase = gamma_lfp[np.where(bin_ind == phase)[0]]
        f_, pxx = sg.welch(strong_theta_atphase, nperseg=1250, noverlap=625, fs=1250)
        df2["freq"] = f_
        df2[str(angle_centers[phase - 1])] = pxx
    bin2Data = bin2Data.append(df2)

    # ----- dividing 360 degress into sliding windows ------------
    window = 40  # degress
    slideby = 5  # degress
    angle_bin = np.arange(0, 360 - 40, slideby)  # divide into 5 bins so each bin=25ms
    angle_centers = angle_bin + window / 2
    bin_ind = np.digitize(theta_angle, bins=angle_bin)
    df3 = pd.DataFrame()
    for phase in angle_bin:
        strong_theta_atphase = gamma_lfp[
            np.where((theta_angle > phase) & (theta_angle < phase + window))[0]
        ]
        f_, pxx = sg.welch(strong_theta_atphase, nperseg=1250, noverlap=625, fs=1250)
        df3["freq"] = f_
        df3[str(phase + window / 2)] = pxx
    slideData = slideData.append(df3)


mean_bin1 = bin1Data.groupby(level=0).mean()
mean_bin2 = bin2Data.groupby(level=0).mean()
mean_slide = slideData.groupby(level=0).mean()

mean_bin1.plot(x="freq", ax=axbin1)
mean_bin2.plot(x="freq", ax=axbin2)
mean_slide.plot(x="freq", ax=axslide, legend=False)

# ---- figure properties ----------
[
    [ax.set_xlabel("Frequency (Hz)"), ax.set_ylabel("Power"), ax.set_xlim([0, 200])]
    for ax in [axbin1, axbin2, axslide]
]
[
    ax.set_title(title)
    for (ax, title) in zip(
        [axbin1, axbin2, axslide], ["5 bins", "9 bins", "sliding window"]
    )
]

# endregion


#%% Multiple regression analysis on slow gamma power explained by variables such as theta-harmonic, theta-asymmetry, speed etc. Also comparing it with theta-harmonic being explained by similar variables
# region

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
    thetalfp = signal_process.filter_sig.bandpass(lfpmaze, lf=1, hf=25)
    hil_theta = signal_process.hilbertfast(thetalfp)
    theta_angle = np.abs(np.angle(hil_theta, deg=True))
    theta_trough = sg.find_peaks(theta_angle)[0]
    theta_peak = sg.find_peaks(-theta_angle)[0]
    theta_amp = np.abs(hil_theta) ** 2

    # --- calculating slow gamma parameters -------
    gammalfp = signal_process.filter_sig.bandpass(lfpmaze, lf=25, hf=50)
    hil_gamma = signal_process.hilbertfast(gammalfp)
    gamma_amp = np.abs(hil_gamma) ** 2

    # --- theta harmonic ----------
    theta_harmonic = signal_process.filter_sig.bandpass(lfpmaze, lf=10, hf=22)
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
    peak_width = (fall_midpoints - rise_midpoints[:-1]) / 1250
    trough_width = (rise_midpoints[1:] - fall_midpoints) / 1250
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

scriptname = os.path.basename(__file__)
filename = "Test"
savefig(fig, filename, scriptname)
