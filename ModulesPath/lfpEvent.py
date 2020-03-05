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


class hwsa(name2path):
    def __init__(self, basePath):
        super().__init__(basePath)

    ##======= hippocampal slow wave detection =============
    def detecthswa(self):
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

        self.delta_epochs = {
            "delta_t": [x[1] for x in delta_epochs],
            "delta_lfp": [x[0] for x in delta_epochs],
        }
        np.save(str(self.filePrefix) + "_hswa.npy", self.delta_epochs)


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

