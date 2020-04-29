import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter

from callfunc import processData

basePath = [
    # "/data/Clustering/SleepDeprivation/RatJ/Day1/",
    # "/data/Clustering/SleepDeprivation/RatK/Day1/",
    "/data/Clustering/SleepDeprivation/RatN/Day1/",
    # "/data/Clustering/SleepDeprivation/RatJ/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day2/",
    "/data/Clustering/SleepDeprivation/RatN/Day2/",
]


sessions = [processData(_) for _ in basePath]

for sub, sess in enumerate(sessions):

    sess.trange = np.array([])
    sess.spikes.extract()
    sess.spikes.stability.firingRate()
    sess.spikes.stability.refPeriodViolation()


ev1, rev1 = sessions[0].replay.expvar()
ev2, rev2 = sessions[1].replay.expvar()
# sess.brainstates.detect()

# violations = sess.spikes.stability.violations

plt.clf()
fig = plt.figure(1, figsize=(6, 10))
gs = GridSpec(1, 2, figure=fig)
fig.subplots_adjust(hspace=0.4)

ax1 = fig.add_subplot(gs[0, 0])
t = (np.linspace(0, 40, 41) * 0.25)[1:] - 0.125
sessions[0].brainstates.addBackgroundtoPlots(ax1)
ax1.fill_between(
    t,
    np.mean(ev1.squeeze(), axis=0) - np.std(ev1.squeeze(), axis=0),
    np.mean(ev1.squeeze(), axis=0) + np.std(ev1.squeeze(), axis=0),
    color="#7c7979",
)
ax1.fill_between(
    t,
    np.mean(rev1.squeeze(), axis=0) - np.std(rev1.squeeze(), axis=0),
    np.mean(rev1.squeeze(), axis=0) + np.std(rev1.squeeze(), axis=0),
    color="#87d498",
)
ax1.plot(t, np.mean(ev1.squeeze(), axis=0), "k")
ax1.plot(t, np.mean(rev1.squeeze(), axis=0), "#02c59b")
ax1.set_xlabel("Time (h)")
ax1.set_ylabel("Explained variance")
ax1.legend(["EV", "REV"])
ax1.text(0.2, 0.28, "POST SD", fontweight="bold")
ax1.set_xlim([0, 10])
ax1.set_ylim([0, 0.3])

ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
sessions[1].brainstates.addBackgroundtoPlots(ax2)
ax2.fill_between(
    t,
    np.mean(ev2.squeeze(), axis=0) - np.std(ev2.squeeze(), axis=0),
    np.mean(ev2.squeeze(), axis=0) + np.std(ev2.squeeze(), axis=0),
    color="#7c7979",
)
ax2.fill_between(
    t,
    np.mean(rev2.squeeze(), axis=0) - np.std(rev2.squeeze(), axis=0),
    np.mean(rev2.squeeze(), axis=0) + np.std(rev2.squeeze(), axis=0),
    color="#87d498",
)
ax2.plot(t, np.mean(ev2.squeeze(), axis=0), "k")
ax2.plot(t, np.mean(rev2.squeeze(), axis=0), "#02c59b")
ax2.set_xlabel("Time (h)")
ax2.text(0.2, 0.28, "POST NSD", fontweight="bold")
ax2.set_xlim([0, 10])
ax2.set_ylim([0, 0.3])
