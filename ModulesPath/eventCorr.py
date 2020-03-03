import numpy as np
import matplotlib.pyplot as plt
from parsePath import name2path
import scipy.signal as sg
import scipy.stats as stat
import pandas as pd
from numpy.fft import fft
import scipy.ndimage as filtSig


class event_event(name2path):

    lfpsRate = 1250
    binSize = 0.250  # in seconds
    nQuantiles = 10

    def __init__(self, basePath):
        super().__init__(basePath)

    def hswa_ripple(self):
        """
        calculating the psth for ripple and slow wave oscillation and making n quantiles for plotting 
        """
        epochs = np.load(str(self.filePrefix) + "_epochs.npy", allow_pickle=True).item()
        pre = epochs["PRE"]  # in seconds
        maze = epochs["MAZE"]  # in seconds
        post = epochs["POST"]  # in seconds
        ripples = np.load(
            str(self.filePrefix) + "_ripples.npy", allow_pickle=True
        ).item()
        ripplesTime = ripples["timestamps"]
        rippleStart = ripplesTime[:, 0] / self.lfpsRate

        delta_epochs = np.load(
            str(self.filePrefix) + "_hswa.npy", allow_pickle=True
        ).item()
        delta_osc = delta_epochs["delta_lfp"]
        delta_t = delta_epochs["delta_t"]

        delta_amp, delta_amp_t = [], []

        for osc, t in zip(delta_osc, delta_t):

            signal = osc
            signal_t = t

            # finding peaks and trough for delta oscillations
            peaks, _ = sg.find_peaks(signal)
            troughs, _ = sg.find_peaks(-signal)

            # making sure they come in pairs and chopping half waves
            if peaks[0] > troughs[0]:
                troughs = troughs[1:]

            if peaks[-1] > troughs[-1]:
                peaks = peaks[:-1]

            for i in range(len(peaks) - 1):
                delta_peak = signal[peaks[i + 1]]
                delta_trough = signal[troughs[i]]

                delta_amp.extend([delta_peak - delta_trough])
                delta_amp_t.append(signal_t[troughs[i]])

        # binning with 500ms before and 1 sec after
        bins = [np.linspace(x - 0.5, x + 1, 150) for x in delta_amp_t]
        t_hist = np.linspace(-0.5, 1, 150)

        ripple_co = [np.histogram(rippleStart, bins=x)[0] for x in bins]
        ripple_co = np.asarray(ripple_co)

        quantiles = pd.qcut(delta_amp, self.nQuantiles, labels=False)

        hwsa_ripple_hist = pd.DataFrame(columns=["name", "quant", "swrs", "time"])
        for category in range(self.nQuantiles):
            indx = np.where(quantiles == category)
            ripple_hist = np.sum(ripple_co[indx], axis=0)
            ripple_hist = filtSig.gaussian_filter1d(ripple_hist, 2)

            temp = pd.DataFrame(
                {"quant": category, "swrs": ripple_hist, "time": t_hist[:-1]}
            )
            hwsa_ripple_hist = hwsa_ripple_hist.append(temp)

        self.hswa_ripple_hist = hwsa_ripple_hist.append(temp)
