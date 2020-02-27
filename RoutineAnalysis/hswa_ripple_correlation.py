import numpy as np
import matplotlib.pyplot as plt
from pfPlot import pf
import pandas as pd
import seaborn as sns
import altair as alt
from lfpEvent import fromLfp
import scipy.signal as sg
from numpy.fft import fft
import scipy.ndimage as filtSig
import matplotlib


# alt.data_transformers.enable("json")

plt.clf()
basePath = [
    "/data/Clustering/SleepDeprivation/RatJ/Day1/",
    "/data/Clustering/SleepDeprivation/RatJ/Day2/",
    "/data/Clustering/SleepDeprivation/RatK/Day1/",
    "/data/Clustering/SleepDeprivation/RatK/Day2/",
    "/data/Clustering/SleepDeprivation/RatN/Day1/",
    "/data/Clustering/SleepDeprivation/RatN/Day2/",
]

sessions = [fromLfp(_) for _ in basePath]


for sess_id, sess in enumerate(sessions):
    sess.hswa()

    epochs = np.load(str(sess.filePrefix) + "_epochs.npy", allow_pickle=True)
    pre = epochs.item().get("PRE")  # in seconds
    maze = epochs.item().get("MAZE")  # in seconds
    post = epochs.item().get("POST")  # in seconds
    ripples = np.load(str(sess.filePrefix) + "_ripples.npy", allow_pickle=True)
    ripplesTime = ripples.item().get("timestamps")
    rippleStart = ripplesTime[:, 0] / 1250

    delta_amp, delta_amp_t = [], []

    for st in range(len(sess.delta_epochs)):

        signal = sess.delta_epochs[st][0]
        signal_t = sess.delta_epochs[st][1]

        if signal_t[0] > post[0] + 5 * 3600:
            spect = fft(signal)
            peaks, _ = sg.find_peaks(signal)
            troughs, _ = sg.find_peaks(-signal)

            if peaks[0] > troughs[0]:
                troughs = troughs[1:]

            if peaks[-1] > troughs[-1]:
                peaks = peaks[:-1]

            for i in range(len(peaks) - 1):
                delta_peak = signal[peaks[i + 1]]
                delta_trough = signal[troughs[i]]

                delta_amp.extend([delta_peak - delta_trough])
                delta_amp_t.append(signal_t[troughs[i]])

    bins = [np.linspace(x - 0.5, x + 1, 150) for x in delta_amp_t]

    ripple_co = [np.histogram(rippleStart, bins=x)[0] for x in bins]
    ripple_co = np.asarray(ripple_co)

    a = pd.qcut(delta_amp, 10, labels=False)

    plt.subplot(3, 2, sess_id + 1)
    cmap = matplotlib.cm.get_cmap("coolwarm")
    for i in range(10):

        indx = np.where(a == i)
        ripple_hist = np.sum(ripple_co[indx], axis=0)
        ripple_hist = filtSig.gaussian_filter1d(ripple_hist, 2)

        t_hist = np.linspace(-0.5, 1, 150)

        plt.plot(t_hist[1:], ripple_hist, color=cmap(i / 10.0))

    plt.xlabel("Time relaive to slow wave (s)")
    plt.ylabel("# SWRs")
    plt.title(sess.subname)

