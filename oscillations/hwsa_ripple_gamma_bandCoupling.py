import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns

from callfunc import processData

basePath = [
    # "/data/Clustering/SleepDeprivation/RatJ/Day1/",
    # "/data/Clustering/SleepDeprivation/RatK/Day1/",
    "/data/Clustering/SleepDeprivation/RatN/Day1/",
    # "/data/Clustering/SleepDeprivation/RatJ/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day2/",
    "/data/Clustering/SleepDeprivation/RatN/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day4/"
]


sessions = [processData(_) for _ in basePath]


group = []
for sub, sess in enumerate(sessions[:3]):

    sess.trange = np.array([])
    t_start = sess.epochs.post[0] + 5 * 3600
    df = sess.brainstates.states
    df = df.loc[(df["state"] == 2) | (df["state"] == 1)]
    df["condition"] = ["sd"] * len(df)
    group.append(df)
    # rem = df.loc[(df["start"] > t_start) & (df["state"] == 2),]
    # nrem = df.loc[(df["start"] > t_start) & (df["state"] == 1),]
    # state_mean = df.groupby(["state"]).mean()
    # state_std = df.groupby(["state"]).std()

    # sd.append(state_mean["duration"])
    # rem =
# sd = np.asarray(sd)
# nsd = []
for sub, sess in enumerate(sessions[3:]):

    sess.trange = np.array([])
    t_start = sess.epochs.post[0]
    df = sess.brainstates.states
    df = df.loc[(df["state"] == 2) | (df["state"] == 1)]

    df["condition"] = ["nsd"] * len(df)
    group.append(df)


group = pd.concat(group, ignore_index=True)
# sd_mean = np.mean(sd, axis=0)
# nsd_mean = np.mean(nsd, axis=0)

# plt.bar([1, 2], [sd_mean[1], nsd_mean[1]])
# plt.errorbar([1, 2], [sd_mean[1], nsd_mean[1]])
# plt.ylim([60, 180])
ax = sns.boxplot(x="state", y="duration", hue="condition", data=group, palette="Set3")
ax.set_ylim(-10, 2000)
ax.set_ylabel("duration (s)")
ax.set_xlabel("")
ax.set_xticklabels(["nrem", "rem"])
#
