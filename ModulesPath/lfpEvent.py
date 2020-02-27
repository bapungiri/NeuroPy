import numpy as np
import matplotlib.pyplot as plt
from parsePath import name2path
import scipy.signal as sg
import scipy.stats as stat
import pandas as pd


class fromLfp(name2path):

    lfpsRate = 1250
    binSize = 0.250  # in seconds

    def __init__(self, basePath):
        super().__init__(basePath)

    def hswa(self):
        """
        fileName: name of the .eeg file
        sampleRate: sampling frequency of eeg
        """

        nyq = 0.5 * self.lfpsRate
        lowdelta = 0.5  # in Hz
        highdelta = 4  # in Hz
        theta = np.load(str(self.filePrefix) + "_BestThetaChan.npy")
        deltastates = np.load(str(self.filePrefix) + "_sws.npy", allow_pickle=True)

        b, a = sg.butter(3, [lowdelta / nyq, highdelta / nyq], btype="bandpass")
        delta_sig = sg.filtfilt(b, a, theta)

        delta = stat.zscore(delta_sig)
        t = np.linspace(0, len(theta) / self.lfpsRate, len(theta))

        states = deltastates.item().get("sws_epochs")

        # delta_epochs = pd.DataFrame(columns=["state", "time", "delta"])
        delta_epochs = []
        for _, st in enumerate(states):
            idx = np.where((t > st[0]) & (t < st[1]))
            delta_st = delta[idx]
            t_st = t[idx]

            # st_data = pd.DataFrame({"state": _, "time": t_st, "delta": delta_st})

            delta_epochs.append([delta_st, t_st])

        self.delta_epochs = delta_epochs

