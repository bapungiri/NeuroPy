import numpy as np
import matplotlib.pyplot as plt
import os
from sleepDetect import SleepScore

basePath = [
    "/data/Clustering/SleepDeprivation/RatJ/Day1/",
    "/data/Clustering/SleepDeprivation/RatJ/Day2/",
    "/data/Clustering/SleepDeprivation/RatK/Day1/",
    "/data/Clustering/SleepDeprivation/RatK/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day3/",
    # "/data/Clustering/SleepDeprivation/RatK/Day4/",
    "/data/Clustering/SleepDeprivation/RatN/Day1/",
    "/data/Clustering/SleepDeprivation/RatN/Day2/",
    # "/data/Clustering/SleepDeprivation/RatN/Day3/",
    # "/data/Clustering/SleepDeprivation/RatN/Day4/",
]

sessions = [SleepScore(_) for _ in basePath]

plt.clf()
for sess in sessions:
    sess.deltaStates()
    epochs = np.load(sess.filePrefix + "_epochs.npy", allow_pickle=True)
    pre = epochs.item().get("PRE")  # in seconds
    maze = epochs.item().get("MAZE")  # in seconds
    post = epochs.item().get("POST")  # in seconds

    states = sess.sws_time
    states_dur = np.diff(states, axis=1)
    states = np.hstack((states, states_dur))
    states = states[(states[:, 0] > post[0]) & (states[:, 2] > 300), :]

    all_sws = []
    t_within = []
    for st in states:
        ind = np.where((sess.delta_t > st[0]) & (sess.delta_t < st[1]))[0]

        raw_sws = sess.delta[ind]
        all_sws.extend(raw_sws)
        x = np.linspace(0, 1, len(raw_sws))
        t_within.extend(x)

        # z = np.polyfit(x, raw_sws, 1)
        # z_fit = z[0] * x + z[1]
        # plt.plot(x, z_fit)

    x = np.linspace(0, 1, 100)
    z = np.polyfit(t_within, all_sws, 1)
    z_fit = z[0] * x + z[1]

    first_last_hour = [np.mean(all_sws[:3600]), np.mean(all_sws[-3600:])]

    plt.plot(first_last_hour, linewidth=2, markersize=12)


plt.legend([sess.subname for sess in sessions])


# plt.plot(all_sws)
# plt.plot(all_sws, ".")
# plt.plot(sess.delta, "k")
# plt.plot(sess.delta_states)
# plt.plot(sess.state_prune[:, 0], np.ones(len(sess.state_prune)), "g.")

