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

from signal_process import filter_sig as filt
from pathlib import Path


class hswa:
    def __init__(self, obj):

        print("hswa running")
        self._obj = obj
        evt = np.load(self._obj.files.slow_wave, allow_pickle=True).item()
        self.amp = evt["delta_amp"]
        self.time = evt["delta_t"]

        if obj.trange.any():
            start, end = obj.trange
            indwhere = np.where((self.time > start) & (self.time < end))
            self.time = self.time[indwhere]
            self.amp = self.amp[indwhere]

    ##======= hippocampal slow wave detection =============
    def detect_hswa(self):
        """
        filters the channel which has highest ripple power.
        Caculate peaks and troughs in the filtered lfp

        ripplechan --> filter delta --> identify peaks and troughs within sws epochs only --> identifies a slow wave as trough to peak --> thresholds for 100ms minimum duration

        """

        # parameters
        min_swa_duration = 0.1  # 100 milliseconds

        # filtering best ripple channel in delta band
        deltachan = ripple.best_chan_lfp
        delta_sig = filt.filter_delta(deltachan)
        delta = stat.zscore(delta_sig)  # normalization w.r.t session

        # epochs which have high slow wave amplitude
        deltastates = np.load(self.f_sws_states, allow_pickle=True).item()
        states = deltastates["sws_epochs"]

        t = np.linspace(0, len(deltachan) / self.lfpsRate, len(deltachan))

        # finding peaks and trough for delta oscillations

        delta_amp, delta_amp_t = [], []
        for epoch in states:
            idx = np.where((t > epoch[0]) & (t < epoch[1]))
            delta_st = delta[idx]
            t_st = t[idx]

            peaks, _ = sg.find_peaks(delta_st)
            troughs, _ = sg.find_peaks(-delta_st)

            # making sure they come in pairs and chopping half waves
            if peaks[0] > troughs[0]:
                troughs = troughs[1:]

            if peaks[-1] > troughs[-1]:
                peaks = peaks[:-1]

            for i in range(len(peaks) - 1):
                swa_peak = delta_st[peaks[i + 1]]
                swa_trough = delta_st[troughs[i]]
                swa_duration = abs(t_st[peaks[i + 1]] - t_st[troughs[i]])

                if swa_duration > min_swa_duration:
                    delta_amp.extend([abs(swa_peak - swa_trough)])
                    delta_amp_t.append(t_st[troughs[i]])

        hipp_slow_wave = {
            "delta_t": np.asarray(delta_amp_t),
            "delta_amp": np.asarray(delta_amp),
        }

        np.save(self.f_slow_wave, hipp_slow_wave)


class ripple:
    def __init__(self, obj):
        self._obj = obj
        print("hello")
        ripple_evt = np.load(obj.files.ripple_evt, allow_pickle=True).item()
        self.time = ripple_evt["timestamps"]
        self.peakpower = ripple_evt["peakPower"]

        if obj.trange.any():
            start, end = obj.trange
            indwhere = np.where((self.time[:, 0] > start) & (self.time[:, 0] < end))
            self.time = self.time[indwhere]
            self.peakpower = self.peakpower[indwhere]

    @property
    def best_chan_lfp(self):
        lfpsrate = self._obj.recinfo.lfpSrate

        lfp = np.load(self._obj.files.ripplelfp, allow_pickle=True).item()
        lfp = lfp["BestChan"]
        lfp_t = np.linspace(0, len(lfp) / lfpsrate, len(lfp))

        if self._obj.trange.any():
            # convert to frames
            start, end = self._obj.trange * lfpsrate
            lfp = lfp[int(start) : int(end)]
            lfp_t = np.linspace(int(start) / lfpsrate, int(end) / lfpsrate, len(lfp))

        return lfp, lfp_t

    def findswr(self):
        ripplechan = self.best_chan_lfp
        ripples = spwrs(ripplechan, self._lfpsRate)

        np.save(self._files.ripple_evt, ripples)
        print(f"{self.f_ripple_evt.name} created")
