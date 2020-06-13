import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.signal as sg
from matplotlib.gridspec import GridSpec
import seaborn as sns
import matplotlib as mpl
from scipy.ndimage import gaussian_filter1d

from callfunc import processData

basePath = [
    "/data/Clustering/SleepDeprivation/RatJ/Day1/",
    "/data/Clustering/SleepDeprivation/RatK/Day1/",
    "/data/Clustering/SleepDeprivation/RatN/Day1/",
    # "/data/Clustering/SleepDeprivation/RatJ/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day2/",
    # "/data/Clustering/SleepDeprivation/RatN/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day4/"
]


sessions = [processData(_) for _ in basePath]

# during REM sleep
plt.close("all")
# fig = plt.figure(1, figsize=(1, 15))
# gs = GridSpec(4, 3, figure=fig)
# fig.subplots_adjust(hspace=0.5)


for sub, sess in enumerate(sessions):

    sess.trange = np.array([])
    tstart = sess.epochs.post[0]
    tend = sess.epochs.post[0] + 5 * 3600

    # sess.spikes.fromCircus("same_folder")
    sess.localsleep.detect(period=[tstart, tend])

    if sub == 2:
        fig = sess.localsleep.plot()

    # fig = sess.localsleep.plot()
    # sess.localsleep.plotAll()

col = ["#FF8F00", "#388E3C", "#9C27B0"]

sd1 = np.zeros(3)
sd5 = np.zeros(3)
ax = fig.add_subplot(3, 5, 12)
for sub, sess in enumerate(sessions):
    tstart = sess.epochs.post[0]
    tend = sess.epochs.post[0] + 5 * 3600

    tbin_offperiods = np.linspace(tstart, tend, 6)
    t_offperiods = sess.localsleep.events.start.values
    hist_off = np.histogram(t_offperiods, bins=tbin_offperiods)[0]
    hist_off = hist_off / 60
    sd1[sub] = hist_off[0]
    sd5[sub] = hist_off[-1]
    # plt.plot(hist_off / 60)

colsub = "#9E9E9E"
ax = fig.add_subplot(3, 5, 11)
ax.plot(np.ones(3), sd1, "o", color=colsub)
ax.plot(3 * np.ones(3), sd5, "o", color=colsub)
ax.plot([1, 3], np.vstack((sd1, sd5)), color=colsub, linewidth=0.8)

mean_grp = np.array([np.mean(sd1), np.mean(sd5)])
sem_grp = np.array([stats.sem(sd1), stats.sem(sd5)])

ax.errorbar(np.array([1, 3]), mean_grp, yerr=sem_grp, color="#263238", fmt="o")
# ax.plot([1, 3], [np.mean(sd1), np.mean(sd5)], color="#263238")
ax.set_xlim([0, 4])
ax.set_ylim([5, 25])
ax.set_xticks([1, 3])
ax.set_xticklabels(["SD1", "SD5"])
ax.set_ylabel("Number per min")

# colsub = "#9E9E9E"
ax = fig.add_subplot(3, 5, 12)

for sub, sess in enumerate(sessions):
    tstart = sess.epochs.post[0]
    tend = sess.epochs.post[0] + 5 * 3600

    fbefore = sess.localsleep.instfiringbefore[:-1].mean(axis=0)
    fbeforestd = sess.localsleep.instfiringbefore[:-1].std(axis=0) / np.sqrt(
        len(sess.localsleep.events)
    )
    fafter = sess.localsleep.instfiringafter[:-1].mean(axis=0)
    fafterstd = sess.localsleep.instfiringafter[:-1].std(axis=0) / np.sqrt(
        len(sess.localsleep.events)
    )
    tbefore = np.linspace(-1, 0, len(fbefore))
    tafter = np.linspace(0.2, 1.2, len(fafter))

    # ax.fill_between(
    #     [0, 0.2],
    #     [min(fbefore), min(fbefore)],
    #     [max(fbefore), max(fbefore)],
    #     color="#BDBDBD",
    #     alpha=0.3,
    # )
    ax.fill_between(
        tbefore, fbefore + fbeforestd, fbefore - fbeforestd, color="#BDBDBD"
    )
    # ax.plot(tbefore, fbefore, color="#616161")
    ax.fill_between(tafter, fafter + fafterstd, fafter - fafterstd, color="#BDBDBD")
    # ax.plot(tafter, fafter, color="#616161")

    # self.events["duration"].plot.kde(ax=ax, color="k")
    # ax.set_xlim([0, max(self.events.duration)])
    ax.set_xlabel("Time from local sleep (s)")
    ax.set_ylabel("Instantneous firing")
    ax.set_xticks([-1, -0.5, 0, 0.2, 0.7, 1.2])
    ax.set_xticklabels(["-1", "-0.5", "start", "end", "0.5", "1"], rotation=45)


ax = fig.add_subplot(3, 5, 13)

for sub, sess in enumerate(sessions):
    tstart = sess.epochs.post[0]
    tend = sess.epochs.post[0] + 5 * 3600
    sess.localsleep.events.duration.plot.kde(color="#BDBDBD")

ax.set_xlim([0, 3])
ax.set_xlabel("Duration (s)")


file = "/data/Clustering/SleepDeprivation/RatN/Day1/RatN_Day1_2019-10-09_03-52-32_localsleep.npy"

data = np.load(file, allow_pickle=True)
