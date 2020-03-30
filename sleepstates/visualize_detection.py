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
gs = GridSpec(5, 1, figure=fig)
fig.subplots_adjust(hspace=0.4)

ax1 = fig.add_subplot(gs[0, 0])
ax1.imshow(
    sess.brainstates.sxx[:, :],
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


fig2 = plt.figure(2)
ax = fig2.add_subplot(111)
ax.scatter(theta_delta_ratio, sess.brainstates.pre_params.emg, s=1)


df = pd.DataFrame({"x": theta_delta_ratio, "y": sess.brainstates.pre_params.emg})
colmap = {1: "r", 2: "g", 3: "b", 4: "k"}
kmeans = KMeans(n_clusters=4)
kmeans.fit(df)
labels = kmeans.predict(df)
centroids = kmeans.cluster_centers_
fig = plt.figure(figsize=(5, 5))

colors = map(lambda x: colmap[x + 1], labels)

plt.scatter(df["x"], df["y"], color=list(colors), alpha=0.5, edgecolor="k")
for idx, centroid in enumerate(centroids):
    plt.scatter(*centroid, color=colmap[idx + 1])
# plt.xlim(0, 80)
# plt.ylim(0, 80)
plt.show()


var = np.asarray(theta_delta_ratio).reshape(-1, 1)

states = hmmfit1d(var)

plt.plot(states)
plt.plot(var)
