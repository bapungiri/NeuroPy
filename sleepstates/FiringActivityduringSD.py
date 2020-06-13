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
fig = plt.figure(1, figsize=(15, 10))
gs = GridSpec(1, 1, figure=fig)
fig.subplots_adjust(hspace=0.5)


for sub, sess in enumerate(sessions):

    sess.trange = np.array([])
    tstart = sess.epochs.post[0]
    tend = sess.epochs.post[0] + 5 * 3600

    # sess.spikes.fromCircus("same_folder")
    # sess.localsleep.detect(period=[tstart, tend])

    # if sub == 2:
    #     fig = sess.localsleep.plot()

    # fig = sess.localsleep.plot()
    # sess.localsleep.plotAll()
    instfiring = sess.localsleep.instfiring
    first_hour = np.median(instfiring[: 3600 * 1000])
    last_hour = np.median(instfiring[-3600 * 1000 :])

    ax = fig.add_subplot(gs[0])

    plt.plot([1, 2], [first_hour, last_hour], "k", alpha=0.5)
    plt.plot([1, 2], [first_hour, last_hour], "o", alpha=0.5)


# data = pd.DataFrame({"ratj": sessions[2].localsleep.instfiring})
# meanfr = data.rolling(3600 * 1000).mean()
# plt.plot(meanfr)
ax.set_xlim([0, 3])
ax.set_xticks([1, 2])
ax.set_xticklabels(["first hour", "last hour"])
ax.set_ylabel("Median inst. firing")
