import numpy as np
from callfunc import processData
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


basePath = [
    # "/data/Clustering/SleepDeprivation/RatJ/Day1/",
    # "/data/Clustering/SleepDeprivation/RatK/Day1/",
    "/data/Clustering/SleepDeprivation/RatN/Day1/",
    # "/data/Clustering/SleepDeprivation/RatJ/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day2/",
    # "/data/Clustering/SleepDeprivation/RatN/Day2/",
]


sessions = [processData(_) for _ in basePath]


for sub, sess in enumerate(sessions):

    # sess.recinfo.makerecinfo()
    sess.trange = np.array([])
    sess.brainstates.detect()

# sess.brainstates.pre_params.hist(bins=400)
# sess.brainstates.pre_params.plot(y="emg")


plt.clf()
fig = plt.figure(1, figsize=(6, 10))
gs = GridSpec(5, 1, figure=fig)
fig.subplots_adjust(hspace=0.4)

ax1 = fig.add_subplot(gs[0, 0])
ax1.imshow(
    sess.brainstates.sxx[:40, :],
    cmap="YlGn",
    aspect="auto",
    # extent=[0, max(t) / 3600, 0, 30.0],
    origin="lower",
    vmin=0,
    vmax=0.2,
    # interpolation="mitchell",
)

ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
ax2.plot(sess.brainstates.pre_params["emg"])

ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
ax3.plot(sess.brainstates.pre_params["theta"])

ax4 = fig.add_subplot(gs[3, 0], sharex=ax1)
ax4.plot(sess.brainstates.pre_params["delta"])
ax4.set_ylim(0, 0.8)

ax5 = fig.add_subplot(gs[4, 0], sharex=ax1)
theta_delta_ratio = (
    sess.brainstates.pre_params["theta"] / sess.brainstates.pre_params["delta"]
)
ax5.plot(theta_delta_ratio)
