import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rnd
import pandas as pd
import pywt
import scipy.fftpack as ft
import scipy.ndimage as smth
import scipy.signal as sg
import scipy.stats as stats
from matplotlib.gridspec import GridSpec

from lfpDetect import swr as spwrs
from signal_process import filter_sig as filt


class hswa:
    def __init__(self, obj):

        self._obj = obj
        if Path(self._obj.sessinfo.files.slow_wave).is_file():
            self._load()

    def _load(self):

        evt = np.load(self._obj.sessinfo.files.slow_wave, allow_pickle=True).item()
        self.amp = np.asarray(evt["delta_amp"])
        self.time = np.asarray(evt["delta_t"])

        if self._obj.trange.any():
            start, end = self._obj.trange
            indwhere = np.where((self.time > start) & (self.time < end))[0]
            self.time = self.time[indwhere]
            self.amp = self.amp[indwhere]

    ##======= hippocampal slow wave detection =============
    def detect_hswa(self):
        """
        filters the channel which has highest ripple power.
        Caculate peaks and troughs in the filtered lfp

        ripplechan --> filter delta --> identify peaks and troughs within sws epochs only --> identifies a slow wave as trough to peak --> thresholds for 100ms minimum duration

        """

        # files
        myinfo = self._obj
        files = self._obj.sessinfo.files

        # parameters
        lfpsRate = myinfo.recinfo.lfpSrate
        min_swa_duration = 0.1  # 100 milliseconds

        # filtering best ripple channel in delta band
        deltachan, t = myinfo.ripple.best_chan_lfp
        delta_sig = filt.filter_delta(deltachan)
        delta = stat.zscore(delta_sig)  # normalization w.r.t session

        # collecting only nrem states
        allstates = myinfo.brainstates.states
        states = allstates[allstates["state"] == 1]

        # t = np.linspace(0, len(deltachan) / lfpsRate, len(deltachan))

        # finding peaks and trough for delta oscillations

        delta_amp, delta_amp_t = [], []
        for epoch in states.itertuples():
            idx = np.where((t > epoch.start) & (t < epoch.end))[0]
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

        np.save(self._obj.sessinfo.files.slow_wave, hipp_slow_wave)

        self._load()


class ripple:
    lowthresholdFactor = 1
    highthresholdFactor = 5
    highRawSigThresholdFactor = 15000
    minRippleDuration = 50  # in milliseconds
    maxRippleDuration = 450  # in milliseconds
    maxRipplePower = 60  # in normalized power units
    mergeDistance = 50

    def __init__(self, obj):

        self._obj = obj

        if Path(self._obj.sessinfo.files.ripple_evt).is_file():

            ripple_evt = np.load(
                obj.sessinfo.files.ripple_evt, allow_pickle=True
            ).item()
            self.time = ripple_evt["timestamps"]
            self.peakpower = ripple_evt["peakPower"]

            if obj.trange.any():
                start, end = obj.trange
                indwhere = np.where((self.time[:, 0] > start) & (self.time[:, 0] < end))
                self.time = self.time[indwhere]
                self.peakpower = self.peakpower[indwhere]

    def best_chan_lfp(self):
        """Returns just best of best channels of each shank or returns all best channels of shanks

        Returns:
            [type] -- [description]
        """
        lfpsrate = self._obj.recinfo.lfpSrate

        lfpinfo = np.load(self._obj.sessinfo.files.ripplelfp, allow_pickle=True).item()
        chans = np.asarray(lfpinfo["channels"])
        lfps = np.asarray(lfpinfo["lfps"])
        coords = lfpinfo["coords"]
        metric = np.asarray(lfpinfo["metricAmp"])

        # lfp_t = np.linspace(0, len(lfps) / lfpsrate, len(lfps))

        # sorting accoriding to the metric values
        descend_indx = np.argsort(metric)[::-1]
        lfps = lfps[descend_indx, :]

        # if chans == "best":
        #     lfps = lfps[0, :]
        #     coords = coords[0]

        # if self._obj.trange.any():
        #     # convert to frames
        #     start, end = self._obj.trange * lfpsrate
        #     lfp = lfp[int(start) : int(end)]
        #     lfp_t = np.linspace(int(start) / lfpsrate, int(end) / lfpsrate, len(lfp))

        return lfps, chans, coords

    def channels(self, viewselection=1):
        """Channels which represent high ripple power in each shank

        """
        sampleRate = self._obj.recinfo.lfpSrate
        duration = 1 * 60 * 60  # chunk of lfp in seconds
        nyq = 0.5 * sampleRate  # Nyquist frequency
        nChans = self._obj.recinfo.nChans
        badchans = self._obj.recinfo.badchans
        allchans = self._obj.recinfo.channels
        changrp = self._obj.recinfo.channelgroups
        nShanks = self._obj.recinfo.nShanks
        probemap = self._obj.recinfo.probemap(probetype="diagbio")
        brainChannels = [item for sublist in changrp[:nShanks] for item in sublist]
        dict_probemap = dict(zip(brainChannels, zip(probemap[0], probemap[1])))

        fileName = self._obj.sessinfo.recfiles.eegfile
        lfpAll = np.memmap(fileName, dtype="int16", mode="r")
        lfpAll = np.memmap.reshape(lfpAll, (int(len(lfpAll) / nChans), nChans))
        lfpCA1 = lfpAll[: sampleRate * duration, :]

        # exclude badchannels
        lfpCA1 = np.delete(lfpCA1, badchans, 1)
        goodChans = np.setdiff1d(allchans, badchans, assume_unique=True)  # keeps order

        # filter and hilbet amplitude for each channel
        avgRipple = np.zeros(lfpCA1.shape[1])
        for i in range(lfpCA1.shape[1]):
            rippleband = filt.filter_ripple(lfpCA1[:, i])
            analytic_signal = sg.hilbert(rippleband)
            amplitude_envelope = np.abs(analytic_signal)
            avgRipple[i] = np.mean(amplitude_envelope, axis=0)

        rippleamp_chan = dict(zip(goodChans, avgRipple))

        # plt.plot(probemap[0], probemap[1], ".", color="#bfc0c0")

        rplchan_shank, lfps, coord, metricAmp = [], [], [], []
        for shank in range(nShanks):
            chans = np.asarray(changrp[shank])
            goodChans_shank = np.setdiff1d(chans, badchans, assume_unique=True)
            avgrpl_shank = np.asarray([rippleamp_chan[key] for key in goodChans_shank])
            chan_max = goodChans_shank[np.argmax(avgrpl_shank)]
            xcoord = dict_probemap[chan_max][0]
            ycoord = dict_probemap[chan_max][1]
            rplchan_shank.append(chan_max)
            lfps.append(lfpAll[:, np.where(goodChans == chan_max)[0][0]])
            coord.append([xcoord, ycoord])
            metricAmp.append(np.max(avgrpl_shank))

        # the reason metricAmp name was used to allow using other metrics such median
        bestripplechans = dict(
            zip(
                ["channels", "lfps", "coords", "metricAmp"],
                [rplchan_shank, lfps, coord, metricAmp],
            )
        )

        filename = self._obj.sessinfo.files.ripplelfp
        np.save(filename, bestripplechans)

    def detect(self):
        """ripples lfp nchans x time

        Returns:
            [type] -- [description]
        """
        ripplelfps, ripplechans, coords = self.best_chan_lfp
        SampFreq = self._obj.recinfo.lfpSrate
        lowFreq = 150
        highFreq = 240
        # TODO chnage raw amplitude threshold to something statistical
        highRawSigThresholdFactor = 15000

        signal = ripplelfps[0, :]
        signal = np.array(signal, dtype=np.float)  # convert data to float
        yf = filt.filter_ripple(signal, ax=-1)

        # hilbert transform --> binarize
        analytic_signal = sg.hilbert(yf, axis=-1)
        amplitude_envelope = np.abs(analytic_signal)
        zscoreSignal = stats.zscore(amplitude_envelope)
        ThreshSignal = np.diff(np.where(zscoreSignal > self.lowthresholdFactor, 1, 0))
        start_ripple = np.argwhere(ThreshSignal == 1)
        stop_ripple = np.argwhere(ThreshSignal == -1)

        firstPass = np.concatenate((start_ripple, stop_ripple), axis=1)

        # TODO delete half ripples in begining or end

        # ===== merging close ripples
        minInterRippleSamples = self.mergeDistance / 1000 * SampFreq
        secondPass = []
        ripple = firstPass[0]
        for i in range(1, len(firstPass)):
            if firstPass[i, 0] - ripple[1] < minInterRippleSamples:
                # Merging ripples
                ripple = [ripple[0], firstPass[i, 1]]
            else:
                secondPass.append(ripple)
                ripple = firstPass[i]

        secondPass.append(ripple)
        secondPass = np.asarray(secondPass)

        # =======delete ripples with less than threshold power
        thirdPass = []
        peakNormalizedPower = []

        for i in range(0, len(secondPass)):
            maxValue = max(zscoreSignal[secondPass[i, 0] : secondPass[i, 1]])
            if maxValue > self.highthresholdFactor:
                thirdPass.append(secondPass[i])
                peakNormalizedPower.append(maxValue)

        thirdPass = np.asarray(thirdPass)

        ripple_duration = np.diff(thirdPass, axis=1) / SampFreq * 1000

        # delete very short ripples
        shortRipples = np.where(ripple_duration < self.minRippleDuration)[0]
        thirdPass = np.delete(thirdPass, shortRipples, 0)
        peakNormalizedPower = np.delete(peakNormalizedPower, shortRipples)
        ripple_duration = np.delete(ripple_duration, shortRipples)

        # delete ripples with unrealistic high power
        # artifactRipples = np.where(peakNormalizedPower > maxRipplePower)[0]
        # fourthPass = np.delete(thirdPass, artifactRipples, 0)
        # peakNormalizedPower = np.delete(peakNormalizedPower, artifactRipples)
        fourthPass = thirdPass

        # delete very long ripples
        veryLongRipples = np.where(ripple_duration > self.maxRippleDuration)[0]
        fifthPass = np.delete(fourthPass, veryLongRipples, 0)
        peakNormalizedPower = np.delete(peakNormalizedPower, veryLongRipples)
        ripple_duration = np.delete(ripple_duration, veryLongRipples)

        # delete ripples which have unusually high amp in raw signal (takes care of disconnection)
        highRawInd = []
        for i in range(0, len(fifthPass)):
            maxValue = max(signal[fifthPass[i, 0] : fifthPass[i, 1]])
            if maxValue > highRawSigThresholdFactor:
                highRawInd.append(i)

        sixthPass = np.delete(fifthPass, highRawInd, 0)
        peakNormalizedPower = np.delete(peakNormalizedPower, highRawInd)

        print(f"{len(sixthPass)} ripples detected")
        # TODO delete sharp ripple like artifacts
        # Maybe unrelistic high power has taken care of this but check to confirm

        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

        ripples = dict()
        ripples["timestamps"] = sixthPass / SampFreq
        ripples["peakPower"] = peakNormalizedPower
        ripples["DetectionParams"] = {
            "lowThres": self.lowthresholdFactor,
            "highThresh": self.highthresholdFactor,
            "ArtifactThresh": self.maxRipplePower,
            "lowFreq": lowFreq,
            "highFreq": highFreq,
            "samplingRate": SampFreq,
            "minDuration": self.minRippleDuration,
            "maxDuration": self.maxRippleDuration,
        }
        ripples["Info"] = {"Date": dt_string}

        # return ripples

        np.save(self._obj.sessinfo.files.ripple_evt, ripples)
        print(f"{self._obj.sessinfo.files.ripple_evt.name} created")

    def plot(self):
        """Gives a comprehensive view of the detection process with some statistics and examples
        """
        _, ripplechans, coords = self.best_chan_lfp()
        probemap = self._obj.recinfo.probemap()
        nChans = self._obj.recinfo.nChans
        changrp = self._obj.recinfo.channelgroups
        chosenShank = changrp[0]
        times = self.time
        peakpower = self.peakpower
        eegfile = self._obj.sessinfo.recfiles.eegfile
        eegdata = np.memmap(eegfile, dtype="int16", mode="r")
        eegdata = np.memmap.reshape(eegdata, (int(len(eegdata) / nChans), nChans))
        eegdata = eegdata[:, chosenShank]

        sort_ind = np.argsort(peakpower)
        peakpower = peakpower[sort_ind]
        times = times[sort_ind, :]
        rpl_duration = np.diff(times, axis=1) * 1000  # in ms
        frames = times * 1250
        nripples = len(peakpower)

        fig = plt.figure(1, figsize=(6, 10))
        gs = GridSpec(4, 10, figure=fig)
        fig.subplots_adjust(hspace=0.6)

        ripple_to_plot = list(range(5)) + list(range(nripples - 5, nripples))
        for ind, ripple in enumerate(ripple_to_plot):
            print(ripple)
            start = int(frames[ripple, 0])
            end = int(frames[ripple, 1])
            lfp = stats.zscore(eegdata[start:end, :])
            ripplebandlfp = filt.filter_ripple(lfp, ax=0)
            # lfp = (lfp.T - np.median(lfp, axis=1)).T
            lfp = lfp + np.linspace(40, 0, lfp.shape[1])
            ripplebandlfp = ripplebandlfp + np.linspace(40, 0, lfp.shape[1])
            duration = (lfp.shape[0] / 1250) * 1000  # in ms

            # if ripple < 5:
            #     row = 1
            # else:
            #     row = 2
            #     ind = ind - 5

            ax = fig.add_subplot(gs[1, ind])
            ax.plot(lfp, "#fa761e", linewidth=0.8)
            ax.set_title(
                f"zsc = {round(peakpower[ripple],2)}, {round(duration)} ms", loc="left"
            )
            ax.set_xlim([0, self.maxRippleDuration / 1000 * 1250])
            ax.axis("off")

            ax = fig.add_subplot(gs[2, ind])
            ax.plot(ripplebandlfp, linewidth=0.8, color="#594f4f")
            # ax.set_title(f"{round(peakpower[ripple],2)}")
            ax.set_xlim([0, self.maxRippleDuration / 1000 * 1250])
            ax.axis("off")

        ax = fig.add_subplot(gs[0, 0])
        ax.text(
            0,
            0.8,
            f" highThresh ={self.highthresholdFactor}\n lowThresh ={self.lowthresholdFactor}\n minDuration = {self.minRippleDuration}\n maxDuration = {self.maxRippleDuration} \n mergeRipple = {self.mergeDistance} \n #Ripples = {len(peakpower)}",
        )
        ax.axis("off")

        ax = fig.add_subplot(gs[0, 1:4])
        ax.plot(probemap[0], probemap[1], ".", color="#cdc6c6")
        ax.plot(coords[0][0], coords[1][1], "r.")
        ax.axis("off")
        ax.set_title("selected channel")

        ax = fig.add_subplot(gs[0, 5])
        histpower, edgespower = np.histogram(peakpower, bins=100)
        ax.plot(edgespower[:-1], histpower, color="#544a4a")
        ax.set_xlabel("Zscore value")
        ax.set_ylabel("Counts")
        # ax.set_yscale("log")

        ax = fig.add_subplot(gs[0, 6])
        histdur, edgesdur = np.histogram(rpl_duration, bins=100)
        ax.plot(edgesdur[:-1], histdur, color="#544a4a")
        ax.set_xlabel("Duration (ms)")
        # ax.set_ylabel("Counts")
        # ax.set_yscale("log")

        subname = self._obj.sessinfo.session.subname
        fig.suptitle(f"Ripple detection of {subname}")


class spindle:
    def __init__(self, obj):
        self._obj = obj

        filename = self._obj.sessinfo.files.spindles
        if filename.is_file():
            spindles = np.load(filename, allow_pickle=True)
            self.time = spindles["timestamps"]
            self.peakpower = spindles["peakPower"]

        else:
            self.time = None
            self.peakpower = None

    def detect(self):
        signal, _ = self._obj.ripple.best_chan_lfp
        SampFreq = self._obj.recinfo.lfpSrate
        nyq = 0.5 * SampFreq
        lowFreq = 8
        highFreq = 16
        lowthresholdFactor = 1.5
        minRippleDuration = 20  # in milliseconds
        maxRippleDuration = 800  # in milliseconds
        maxRipplePower = 60  # in normalized power units

        signal = filt.filter_spindle(signal)
        zsc_sig = stats.zscore(signal)
        analytic_signal = sg.hilbert(zsc_sig)
        amplitude_envelope = np.abs(analytic_signal)

        plt.plot(amplitude_envelope)

    def plot(self):
        pass
