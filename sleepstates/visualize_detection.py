import numpy as np
from callfunc import processData
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.cluster import KMeans
from hmmlearn.hmm import GaussianHMM

import pandas as pd


def hmmfit1d(Data):
    # hmm states on 1d data and returns labels with highest mean = highest label
    model = GaussianHMM(n_components=2, n_iter=100).fit(Data)
    hidden_states = model.predict(Data)
    mus = np.squeeze(model.means_)
    sigmas = np.squeeze(np.sqrt(model.covars_))
    transmat = np.array(model.transmat_)

    idx = np.argsort(mus)
    mus = mus[idx]
    sigmas = sigmas[idx]
    transmat = transmat[idx, :][:, idx]

    state_dict = {}
    states = [i for i in range(4)]
    for i in idx:
        state_dict[idx[i]] = states[i]

    relabeled_states = [state_dict[h] for h in hidden_states]
    # relabeled_states = hidden_states
    return relabeled_states


basePath = [
    # "/data/Clustering/SleepDeprivation/RatJ/Day1/",
    # "/data/Clustering/SleepDeprivation/RatK/Day1/",
    # "/data/Clustering/SleepDeprivation/RatN/Day1/",
    # "/data/Clustering/SleepDeprivation/RatJ/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day2/",
    "/data/Clustering/SleepDeprivation/RatN/Day2/"
]


sessions = [processData(_) for _ in basePath]


for sub, sess in enumerate(sessions):

    # sess.recinfo.makerecinfo()
    sess.trange = np.array([])
    sess.brainstates.detect()

# sess.brainstates.pre_params.hist(bins=400)
# sess.brainstates.pre_params.plot(y="emg")
plt.close("all")
plt.clf()
fig = plt.figure(1, figsize=(6, 10))
gs = GridSpec(6, 1, figure=fig)
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


ax6 = fig.add_subplot(gs[5, 0], sharex=ax1)

states = sess.brainstates.pre_params["state"]
x = np.arange(0, len(states))

nrem = np.where(states == 1, 1, 0)
ax6.fill_between(x, nrem, 0, color="#6b90d1")

rem = np.where(states == 2, 1, 0)
ax6.fill_between(x, rem, 0, color="#eb9494")

qw = np.where(states == 3, 1, 0)
ax6.fill_between(x, qw, 0, color="#b6afaf")

active = np.where(states == 4, 1, 0)
ax6.fill_between(x, active, 0, color="#201d1d")
