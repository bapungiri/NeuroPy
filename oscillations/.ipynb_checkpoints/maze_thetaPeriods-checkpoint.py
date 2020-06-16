import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as sg
import scipy.stats as stats
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button, RadioButtons, Slider
from scipy.ndimage import gaussian_filter, gaussian_filter1d
import ipywidgets as widgets

from callfunc import processData
from signal_process import filter_sig, hilbertfast, wavelet_decomp

#%%
cmap = mpl.cm.get_cmap("hot_r")

basePath = [
    # "/data/Clustering/SleepDeprivation/RatJ/Day1/",
    # "/data/Clustering/SleepDeprivation/RatK/Day1/",
    "/data/Clustering/SleepDeprivation/RatN/Day1/",
    # "/data/Clustering/SleepDeprivation/RatJ/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day2/",
    # "/data/Clustering/SleepDeprivation/RatN/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day4/"
]


sessions = [processData(_) for _ in basePath]

plt.clf()
fig = plt.figure(1, figsize=(1, 15))
gs = GridSpec(6, 4, figure=fig)
fig.subplots_adjust(hspace=0.5)


def doWavelet(lfp, freqs, ncycles=3):
    wavdec = wavelet_decomp(lfp, freqs=freqs)
    wav = wavdec.cohen(ncycles=ncycles)
    wav = stats.zscore(wav)
    wav = gaussian_filter(wav, sigma=4)

    return wav


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

        axgamma = fig.add_subplot(gs[0, :])
        axgamma.clear()
        frgamma = np.arange(25, 50, 1)
        wav = doWavelet(lfpmaze, freqs=frgamma, ncycles=7)
        axgamma.pcolorfast(tmaze, frgamma, wav, cmap="jet")
        axgamma.set_xlim([tstart, tend])
        axgamma.set_ylabel("Frequency")

        axtheta = fig.add_subplot(gs[1, :], sharex=axgamma)
        axtheta.clear()
        frtheta = np.arange(2, 20, 0.5)
        wav = doWavelet(lfpmaze, freqs=frtheta, ncycles=3)
        axtheta.pcolorfast(tmaze, frtheta, wav, cmap="jet")
        axtheta.set_ylabel("Amplitude")

        axpos = fig.add_subplot(gs[4:6, 0])
        colpos = [cmap(_ / len(posmazex)) for _ in range(len(posmazex))]
        axpos.clear()
        axpos.plot(posx, posy, color="#bfc0c0", zorder=1)
        # axpos.plot(postmaze, posmazex, "k")
        # axpos.plot(postmaze, posmazey, "r")
        axpos.scatter(posmazex, posmazey, s=3, zorder=2, c=colpos)
        axpos.axis("off")

        axspeed = fig.add_subplot(gs[2, :], sharex=axgamma)
        axspeed.clear()
        # axspeed.plot(postmaze[:-1], np.abs(np.diff(posmazex)), "k")
        # axspeed.plot(postmaze[:-1], np.abs(np.diff(posmazey)), "r")
        axspeed.plot(postmaze[:-1], speed, "k")
        axspeed.set_ylabel("Speed (a.u.)")

        axlfp = fig.add_subplot(gs[3, :], sharex=axgamma)
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


#%%
