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
gs = gridspec.GridSpec(4, 4, figure=fig)
fig.subplots_adjust(hspace=0.3, wspace=0.3)
fig.suptitle("Slow gamma related")
titlesize = 8


# %% Detects strong theta within maze and averages spectrogram around theta cycles using Wavelets
# region
inner = gridspec.GridSpecFromSubplotSpec(
    2, 1, subplot_spec=gs[0, 0], wspace=0.1, hspace=0.1
)
cmap = mpl.cm.get_cmap("hot_r")
strong_theta = []
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
    wav = wavdec.cohen()
    # frgamma = np.arange(25, 50, 1)
    # wavdec = wavelet_decomp(lfpmaze, freqs=frgamma)
    # wav = wavdec.colgin2009()
    # wavtheta = doWavelet(lfpmaze, freqs=frtheta, ncycles=3)

    sum_theta = gaussian_filter1d(np.sum(wav, axis=0), sigma=10)
    zsc_theta = stats.zscore(sum_theta)
    thetaevents = threshPeriods(
        zsc_theta, lowthresh=0, highthresh=1.5, minDistance=300, minDuration=1250
    )

    # strong_theta = []
    theta_indices = []
    for (beg, end) in thetaevents:
        strong_theta.extend(lfpmaze[beg:end])
        theta_indices.extend(np.arange(beg, end))
    strong_theta.extend(np.asarray(strong_theta))
    # theta_indices = np.asarray(theta_indices)
    # non_theta = np.delete(lfpmaze, theta_indices)


frgamma = np.arange(25, 150, 1)
# frgamma = np.linspace(25, 150, 1)
strong_theta = np.asarray(strong_theta)
wavdec = signal_process.wavelet_decomp(strong_theta, freqs=frgamma)
wav = wavdec.colgin2009()
# wav = wavdec.cohen(ncycles=7)

wav = stats.zscore(wav, axis=1)

theta_filter = stats.zscore(
    signal_process.filter_sig.filter_cust(strong_theta, lf=4, hf=11)
)

hil_theta = signal_process.hilbertfast(theta_filter)
theta_amp = np.abs(hil_theta)
theta_angle = np.angle(hil_theta, deg=True) + 180
theta_troughs = sg.find_peaks(-theta_filter)[0]

bin_angle = np.linspace(0, 360, int(360 / 9) + 1)
bin_ind = np.digitize(theta_angle, bin_angle)

wav_phase = []
for i in np.unique(bin_ind):
    find_where = np.where(bin_ind == i)[0]
    wav_at_angle = np.mean(wav[:, find_where], axis=1)
    wav_phase.append(wav_at_angle)

wav_phase = np.asarray(wav_phase).T

ax = fig.add_subplot(inner[:, :])
ax.pcolorfast(bin_angle[:-1], frgamma[:-1], wav_phase, cmap="Spectral_r")
ax.set_xlabel(r"$\theta$ phase")
ax.set_ylabel("frequency (Hz)")

bicoh, freq, bispec = signal_process.bicoherence(strong_theta, fhigh=100)

# bicoh = gaussian_filter(bicoh, sigma=2)
# bicoh = np.where(bicoh > 0.05, bicoh, 0)
bispec_real = gaussian_filter(np.real(bispec), sigma=2)
bispec_imag = gaussian_filter(np.imag(bispec), sigma=2)
bispec_angle = gaussian_filter(np.angle(bispec, deg=True), sigma=2)

ax = fig.add_subplot(gs[0, 1])
ax.clear()
im = ax.pcolorfast(freq, freq, bicoh, cmap="Spectral_r", vmax=0.05, vmin=0)
# ax.contour(freq, freq, bicoh, levels=[0.1, 0.2, 0.3], colors="k", linewidths=1)
ax.set_ylim([1, max(freq) / 2])
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Frequency (Hz)")


# cax = fig.add_axes([0.3, 0.8, 0.5, 0.05])
# cax.clear()
# ax.contour(freq, freq, bicoh, levels=[0.1, 0.2, 0.3], colors="k", linewidths=1)
fig.colorbar(im, ax=ax, orientation="horizontal")
# avg_theta = np.zeros(156)
# mean_gamma = np.zeros((wav.shape[0], 156))
# for i in theta_troughs[1:]:
#     mean_gamma = mean_gamma + wav[:, i - 125 : i + 31]
#     avg_theta = avg_theta + strong_theta[i - 125 : i + 31]

# mean_gamma = mean_gamma / len(theta_troughs)
# mean_gamma = stats.zscore(mean_gamma, axis=1)


# ax = fig.add_subplot(inner[0])
# t_thetacycle = np.linspace(-100, 25, 156)
# ax.pcolorfast(t_thetacycle, frgamma, mean_gamma, cmap="Spectral")
# ax.set_ylabel("Frequency (Hz)")

# # ax.contourf(t_thetacycle,frgamma,mean_gamma)
# ax.set_xlim([-100, 25])
# ax = fig.add_subplot(inner[1], sharex=ax)
# ax.plot(t_thetacycle, avg_theta / len(theta_troughs), "k")
# ax.set_xlabel("Time from theta trough (ms)")
# ax.set_ylabel("Amplitude")

# endregion
