import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rnd
import pandas as pd
import scipy.fftpack as ft
import scipy.ndimage as smth
import scipy.signal as sg
import scipy.stats as stat

from lfpDetect import swr as spwrs
from parsePath import name2path


class hswa(name2path):
    def __init__(self, basePath):
        super().__init__(basePath)

    ##======= hippocampal slow wave detection =============
    def detect_hswa(self):
        """
        fileName: name of the .eeg file
        sampleRate: sampling frequency of eeg
        """

        nyq = 0.5 * self.lfpsRate
        lowdelta = 0.5  # in Hz
        highdelta = 4  # in Hz
        deltachan = np.load(self.f_ripplelfp, allow_pickle=True).item()
        deltachan = deltachan["BestChan"]
        deltastates = np.load(str(self.filePrefix) + "_sws.npy", allow_pickle=True)

        b, a = sg.butter(3, [lowdelta / nyq, highdelta / nyq], btype="bandpass")
        delta_sig = sg.filtfilt(b, a, deltachan)

        delta = stat.zscore(delta_sig)
        t = np.linspace(0, len(deltachan) / self.lfpsRate, len(deltachan))

        states = deltastates.item().get("sws_epochs")

        # delta_epochs = pd.DataFrame(columns=["state", "time", "delta"])
        delta_epochs = []
        for _, st in enumerate(states):
            idx = np.where((t > st[0]) & (t < st[1]))
            delta_st = delta[idx]
            t_st = t[idx]

            # st_data = pd.DataFrame({"state": _, "time": t_st, "delta": delta_st})

            delta_epochs.append([delta_st, t_st])

        # self.delta_epochs = {
        #     "delta_t": [x[1] for x in delta_epochs],
        #     "delta_lfp": [x[0] for x in delta_epochs],
        # }
        # np.save(str(self.filePrefix) + "_hswa.npy", self.delta_epochs)

        delta_t = [x[1] for x in delta_epochs]
        delta_osc = [x[0] for x in delta_epochs]

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

        hipp_slow_wave = {"delta_t": delta_amp_t, "delta_amp": delta_amp}

        np.save(self.f_slow_wave, hipp_slow_wave)


class swr(name2path):
    def __init__(self, basePath):
        super().__init__(basePath)

    def findswr(self):
        ripplechan = np.load(self.f_ripplelfp, allow_pickle=True).item()
        ripplechan = ripplechan["BestChan"]
        self.__ripples = spwrs(ripplechan, self.lfpsRate)

        np.save(self.f_ripple_evt, self.__ripples)
        print(f"{self.f_ripple_evt.name} created")

    def load_swr_evt(self):
        self.ripples = np.load(self.f_ripple_evt, allow_pickle=True).item()

        # self.peakripplePower = self.ripples["peakPower"]
        print(f"ripple events loaded .....")
        # print(f"following keys {list(self.ripples.keys())} can be used")

