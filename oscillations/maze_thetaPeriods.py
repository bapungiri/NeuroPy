import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.signal as sg
from matplotlib.gridspec import GridSpec
import seaborn as sns
from signal_process import filter_sig, hilbertfast
from signal_process import wavelet_decomp
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from matplotlib.widgets import Slider, Button, RadioButtons

from callfunc import processData

basePath = [
    # "/data/Clustering/SleepDeprivation/RatJ/Day1/",
    # "/data/Clustering/SleepDeprivation/RatK/Day1/",
    # "/data/Clustering/SleepDeprivation/RatN/Day1/",
    # "/data/Clustering/SleepDeprivation/RatJ/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day2/",
    "/data/Clustering/SleepDeprivation/RatN/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day4/"
]


sessions = [processData(_) for _ in basePath]

plt.clf()
fig = plt.figure(1, figsize=(1, 15))
gs = GridSpec(5, 3, figure=fig)
fig.subplots_adjust(hspace=0.5)

colband = ["#CE93D8", "#1565C0", "#E65100"]

for sub, sess in enumerate(sessions):

    sess.trange = np.array([])
    posx = sess.position.x
    posy = sess.position.y
    post = sess.position.t
    tstart = sess.epochs.maze[0]
    tend = sess.epochs.maze[1]
    lfp, _, _ = sess.spindle.best_chan_lfp()
    t = np.linspace(0, len(lfp) / 1250, len(lfp))

    lfpmaze = lfp[(t > tstart) & (t < tend)]
    tmaze = np.linspace(tstart, tend, len(lfpmaze))
    posmazex = posx[(post > tstart) & (post < tend)]
    posmazey = posy[(post > tstart) & (post < tend)]
    postmaze = np.linspace(tstart, tend, len(posmazex))
    speed = np.sqrt(np.diff(posmazex) ** 2 + np.diff(posmazey) ** 2)
    speed = gaussian_filter1d(speed, sigma=10)

    ax = fig.add_subplot(gs[0, :])
    freqs = np.arange(20, 50, 2)
    wavdec = wavelet_decomp(lfpmaze, freqs=freqs)
    wav = wavdec.cohen(ncycles=7)
    wav = stats.zscore(wav)
    wav = gaussian_filter(wav, sigma=4)
    ax.pcolorfast(tmaze, freqs, wav, cmap="jet")

    del wav

    ax = fig.add_subplot(gs[1, :], sharex=ax)
    freqs = np.arange(2, 12, 1)
    wavdec = wavelet_decomp(lfpmaze, freqs=freqs)
    wav = wavdec.cohen()
    wav = stats.zscore(wav)
    wav = gaussian_filter(wav, sigma=4)
    ax.pcolorfast(tmaze, freqs, wav, cmap="jet")
    # ax.imshow(wav, cmap="jet", aspect="auto", vmax=3)

    ax = fig.add_subplot(gs[2, :], sharex=ax)
    ax.plot(postmaze, posmazex, "k")
    ax.plot(postmaze, posmazey, "r")

    ax = fig.add_subplot(gs[3, :], sharex=ax)
    # ax.plot(postmaze[:-1], np.abs(np.diff(posmazex)), "k")
    # ax.plot(postmaze[:-1], np.abs(np.diff(posmazey)), "r")
    ax.plot(postmaze[:-1], speed, "k")

    ax = fig.add_subplot(gs[4, :], sharex=ax)
    ax.plot(tmaze, lfpmaze, "k")


fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)
t = np.arange(0.0, 1.0, 0.001)
a0 = 5
f0 = 3
delta_f = 5.0
s = a0 * np.sin(2 * np.pi * f0 * t)
(l,) = plt.plot(t, s, lw=2)
ax.margins(x=0)

axcolor = "lightgoldenrodyellow"
axfreq = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
axamp = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)

sfreq = Slider(axfreq, "Freq", 0.1, 30.0, valinit=f0, valstep=delta_f)
samp = Slider(axamp, "Amp", 0.1, 10.0, valinit=a0)


def update(val):
    amp = samp.val
    freq = sfreq.val
    l.set_ydata(amp * np.sin(2 * np.pi * freq * t))
    fig.canvas.draw_idle()


sfreq.on_changed(update)
samp.on_changed(update)
