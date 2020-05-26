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
from scipy import fftpack
from lfpDetect import swr as spwrs
from signal_process import filter_sig as filt


class Hswa:
    def __init__(self, obj):

        self._obj = obj
        if Path(self._obj.sessinfo.files.slow_wave).is_file():
            self._load()

    def _load(self):

        evt = np.load(self._obj.sessinfo.files.slow_wave, allow_pickle=True).item()
        self.peakamp = np.asarray(evt["peakamp"])
        self.time = np.asarray(evt["time"])
        self.tbeg = np.asarray(evt["tbeg"])
        self.tend = np.asarray(evt["tend"])
        self.endamp = np.asarray(evt["endamp"])

        # if self._obj.trange.any():
        #     start, end = self._obj.trange
        #     indwhere = np.where((self.time > start) & (self.time < end))[0]
        #     self.time = self.time[indwhere]
        #     self.amp = self.amp[indwhere]

    ##======= hippocampal slow wave detection =============
    def detect(self):
        """
        filters the channel which has highest ripple power.
        Caculate peaks and troughs in the filtered lfp

        ripplechan --> filter delta --> identify peaks and troughs within sws epochs only --> identifies a slow wave as trough to peak --> thresholds for 100ms minimum duration

        """

        # files
        myinfo = self._obj
        files = self._obj.sessinfo.files

        # parameters
        lfpsRate = self._obj.recinfo.lfpSrate
        min_swa_duration = 0.1  # 100 milliseconds

        # filtering best ripple channel in delta band
        deltachan, _, _ = myinfo.spindle.best_chan_lfp()
        t = np.linspace(0, len(deltachan) / lfpsRate, len(deltachan))
        delta_sig = filt.filter_delta(deltachan, ax=-1)
        delta = stats.zscore(delta_sig)  # normalization w.r.t session
        delta = -delta  # flipping as this is in sync with cortical slow wave

        # collecting only nrem states
        allstates = myinfo.brainstates.states
        states = allstates[allstates["name"] == "nrem"]

        # finding peaks and trough for delta oscillations
        sigdelta = []
        for epoch in states.itertuples():
            idx = np.where((t > epoch.start) & (t < epoch.end))[0]
            delta_st = delta[idx]
            t_st = t[idx]

            grad = np.gradient(delta_st)
            zero_crossings = np.where(np.diff(np.sign(grad)))[0]
            cross_sign = np.zeros(len(zero_crossings))

            for i, ind in enumerate(zero_crossings):
                if grad[ind - 1] < grad[ind + 1]:
                    cross_sign[i] = 1

            up = zero_crossings[np.where(cross_sign == 1)[0]]
            down = zero_crossings[np.where(cross_sign == 0)[0]]

            if down[0] < up[0]:
                down = down[1:]
            if down[-1] > up[-1]:
                down = down[:-1]

            sigdelta = []
            for i in range(len(up) - 1):
                tbeg = t_st[up[i]]
                tpeak = t_st[down[i]]
                tend = t_st[up[i + 1]]
                peakamp = delta_st[down[i]]
                endamp = delta_st[up[i + 1]]
                if (peakamp > 2 and endamp < 0) or (peakamp > 1 and endamp < -1.5):
                    sigdelta.append([peakamp, endamp, tpeak, tbeg, tend])

        sigdelta = np.asarray(sigdelta)
        print(f"{len(sigdelta)} delta detected")

        hipp_slow_wave = {
            "time": sigdelta[:, 2],
            "tbeg": sigdelta[:, 3],
            "tend": sigdelta[:, 4],
            "peakamp": sigdelta[:, 0],
            "endamp": sigdelta[:, 1],
        }

        np.save(self._obj.sessinfo.files.slow_wave, hipp_slow_wave)

        self._load()

    def plot(self):
        """Gives a comprehensive view of the detection process with some statistics and examples
        """
        _, spindlechan, coord = self._obj.spindle.best_chan_lfp()
        eegSrate = self._obj.recinfo.lfpSrate
        probemap = self._obj.recinfo.probemap()
        nChans = self._obj.recinfo.nChans
        changrp = self._obj.recinfo.channelgroups
        badchans = self._obj.recinfo.badchans
        chosenShank = changrp[0]
        chosenShank = np.setdiff1d(np.array(chosenShank), badchans)
        times = self.time
        tbeg = self.tbeg
        tend = self.tend
        eegfile = self._obj.sessinfo.recfiles.eegfile
        eegdata = np.memmap(eegfile, dtype="int16", mode="r")
        eegdata = np.memmap.reshape(eegdata, (int(len(eegdata) / nChans), nChans))
        eegdata = eegdata[:, chosenShank]

        # sort_ind = np.argsort(peakpower)
        # peakpower = peakpower[sort_ind]
        # times = times[sort_ind, :]
        # rpl_duration = np.diff(times, axis=1) * 1000  # in ms
        frames = times * eegSrate
        framesbeg = tbeg * eegSrate
        framesend = tend * eegSrate
        ndelta = len(times)

        fig = plt.figure(1, figsize=(6, 10))
        gs = GridSpec(2, 10, figure=fig)
        fig.subplots_adjust(hspace=0.2)

        delta_to_plot = list(range(50, 60))

        beg_eeg = int(framesbeg[delta_to_plot[0]]) - eegSrate
        end_eeg = int(framesend[delta_to_plot[-1]]) + eegSrate
        lfp = stats.zscore(eegdata[beg_eeg:end_eeg, :])
        lfp = lfp + np.linspace(40, 0, lfp.shape[1])
        eegt = np.linspace(beg_eeg, end_eeg, len(lfp))
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(eegt, lfp, "#444040", linewidth=0.8)
        ax1.set_title("Raw lfp", loc="left")

        for ind, delta in enumerate(delta_to_plot):
            start = int(framesbeg[delta])
            peak = int(frames[delta])
            end = int(framesend[delta])
            ax1.plot([peak, peak], [-8, 47], "--")
            ax1.fill_between([start, end], [-6, -6], [45, 45], alpha=0.3)
            ax1.axis("off")

        deltabandlfp = filt.filter_delta(lfp, ax=0)
        deltabandlfp = deltabandlfp + np.linspace(40, 0, lfp.shape[1])
        ax2 = fig.add_subplot(gs[1, :])
        ax2.plot(eegt, deltabandlfp, "#444040", linewidth=0.8)
        ax2.set_title("Filtered lfp", loc="left")
        for ind, delta in enumerate(delta_to_plot):
            start = int(framesbeg[delta])
            peak = int(frames[delta])
            end = int(framesend[delta])
            ax2.plot([peak, peak], [-8, 47], "--")
            ax2.fill_between([start, end], [-6, -6], [45, 45], alpha=0.3)
            ax2.axis("off")

        subname = self._obj.sessinfo.session.subname
        fig.suptitle(f"Delta wave detection of {subname}")


class Ripple:
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
            self.time = ripple_evt["times"]
            self.peakpower = ripple_evt["peakPower"]
            self.peaktime = ripple_evt["peaktime"]

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
        nChans = self._obj.recinfo.nChans
        badchans = self._obj.recinfo.badchans
        allchans = self._obj.recinfo.channels
        changrp = self._obj.recinfo.channelgroups
        nShanks = self._obj.recinfo.nShanks
        probemap = self._obj.recinfo.probemap()
        brainChannels = [item for sublist in changrp[:nShanks] for item in sublist]
        dict_probemap = dict(zip(brainChannels, zip(probemap[0], probemap[1])))

        fileName = self._obj.sessinfo.recfiles.eegfile
        lfpAll = np.memmap(fileName, dtype="int16", mode="r")
        lfpAll = np.memmap.reshape(lfpAll, (int(len(lfpAll) / nChans), nChans))
        lfpCA1 = lfpAll[: sampleRate * duration, :]

        # exclude badchannels
        lfpCA1 = np.delete(lfpCA1, badchans, 1)
        goodChans = np.setdiff1d(allchans, badchans, assume_unique=True)  # keeps order

        hilbertfast = lambda x: sg.hilbert(x, fftpack.next_fast_len(len(x)))[: len(x)]

        # filter and hilbet amplitude for each channel
        avgRipple = np.zeros(lfpCA1.shape[1])
        for i in range(lfpCA1.shape[1]):
            rippleband = filt.filter_ripple(lfpCA1[:, i])
            amplitude_envelope = np.abs(hilbertfast(rippleband))
            avgRipple[i] = np.mean(amplitude_envelope, axis=0)

        rippleamp_chan = dict(zip(goodChans, avgRipple))

        # plt.plot(probemap[0], probemap[1], ".", color="#bfc0c0")

        rplchan_shank, lfps, coord, metricAmp = [], [], [], []
        for shank in range(nShanks):
            chans = np.asarray(changrp[shank])
            goodChans_shank = np.setdiff1d(chans, badchans, assume_unique=True)

            if goodChans_shank.size != 0:
                avgrpl_shank = np.asarray(
                    [rippleamp_chan[key] for key in goodChans_shank]
                )
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

    def _zscbestchannel(self, fromfile=1):
        filename = str(self._obj.sessinfo.files.filePrefix) + "_zscbestRippleChan.npy"

        if fromfile == 1:
            signal = np.load(filename)
        else:
            ripplelfps, _, _ = self.best_chan_lfp()
            signal = np.asarray(ripplelfps, dtype=np.float)  # convert data to float

            # optimizing memory and performance writing on the same array of signal
            hilbertfast = lambda x: sg.hilbert(x, fftpack.next_fast_len(len(x)))[
                : len(x)
            ]

            for i in range(signal.shape[0]):
                print(i)
                yf = filt.filter_ripple(signal[i, :], ax=-1)
                # amplitude_envelope = np.abs(sg.hilbert(yf, axis=-1))
                amplitude_envelope = np.abs(hilbertfast(yf))
                signal[i, :] = stats.zscore(amplitude_envelope, axis=-1)

            np.save(filename, signal)

        return signal

    def detect(self):
        """ripples lfp nchans x time

        Returns:
            [type] -- [description]
        """
        SampFreq = self._obj.recinfo.lfpSrate
        lowFreq = 150
        highFreq = 240
        # TODO chnage raw amplitude threshold to something statistical
        highRawSigThresholdFactor = 15000

        ripplelfps, _, _ = self.best_chan_lfp()
        signal = np.asarray(ripplelfps, dtype=np.float)[0, :]

        zscsignal = self._zscbestchannel(fromfile=0)

        # delete ripples in noisy period
        deadfile = (self._obj.sessinfo.files.filePrefix).with_suffix(".dead")
        if deadfile.is_file():
            with deadfile.open("r") as f:
                noisy = []
                for line in f:
                    epc = line.split(" ")
                    epc = [float(_) for _ in epc]
                    noisy.append(epc)
                noisy = np.asarray(noisy)
                noisy = ((noisy / 1000) * SampFreq).astype(int)

            for noisy_ind in range(noisy.shape[0]):
                st = noisy[noisy_ind, 0]
                en = noisy[noisy_ind, 1]
                numnoisy = en - st
                zscsignal[:, st:en] = np.zeros((zscsignal.shape[0], numnoisy))

        # hilbert transform --> binarize by > than lowthreshold
        maxPower = np.max(zscsignal, axis=0)
        ThreshSignal = np.where(zscsignal > self.lowthresholdFactor, 1, 0).sum(axis=0)
        ThreshSignal = np.diff(np.where(ThreshSignal > 0, 1, 0))
        start_ripple = np.where(ThreshSignal == 1)[0]
        stop_ripple = np.where(ThreshSignal == -1)[0]

        if start_ripple[0] > stop_ripple[0]:
            stop_ripple = stop_ripple[1:]
        if start_ripple[-1] > stop_ripple[-1]:
            start_ripple = start_ripple[:-1]

        firstPass = np.vstack((start_ripple, stop_ripple)).T
        print(f"{len(firstPass)} ripples detected initially")

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
        print(f"{len(secondPass)} ripples reamining after merging")

        # =======delete ripples with less than threshold power
        thirdPass = []
        peakNormalizedPower, peaktime = [], []

        for i in range(0, len(secondPass)):
            maxValue = max(maxPower[secondPass[i, 0] : secondPass[i, 1]])
            if maxValue > self.highthresholdFactor:
                thirdPass.append(secondPass[i])
                peakNormalizedPower.append(maxValue)
                peaktime.append(
                    [
                        secondPass[i, 0]
                        + np.argmax(maxPower[secondPass[i, 0] : secondPass[i, 1]])
                    ]
                )
        thirdPass = np.asarray(thirdPass)
        print(f"{len(thirdPass)} ripples reamining after deleting weak ripples")

        ripple_duration = np.diff(thirdPass, axis=1) / SampFreq * 1000

        # delete very short ripples
        shortRipples = np.where(ripple_duration < self.minRippleDuration)[0]
        fourthPass = np.delete(thirdPass, shortRipples, 0)
        peakNormalizedPower = np.delete(peakNormalizedPower, shortRipples)
        peaktime = np.delete(peaktime, shortRipples)
        ripple_duration = np.delete(ripple_duration, shortRipples)
        print(f"{len(fourthPass)} ripples reamining after deleting short ripples")

        # delete ripples with unrealistic high power
        # artifactRipples = np.where(peakNormalizedPower > maxRipplePower)[0]
        # fourthPass = np.delete(thirdPass, artifactRipples, 0)
        # peakNormalizedPower = np.delete(peakNormalizedPower, artifactRipples)

        # delete very long ripples
        veryLongRipples = np.where(ripple_duration > self.maxRippleDuration)[0]
        fifthPass = np.delete(fourthPass, veryLongRipples, 0)
        peakNormalizedPower = np.delete(peakNormalizedPower, veryLongRipples)
        peaktime = np.delete(peaktime, veryLongRipples)
        ripple_duration = np.delete(ripple_duration, veryLongRipples)
        print(f"{len(fifthPass)} ripples reamining after deleting very long ripples")

        # delete ripples which have unusually high amp in raw signal (takes care of disconnection)
        highRawInd = []
        for i in range(0, len(fifthPass)):
            maxValue = max(signal[fifthPass[i, 0] : fifthPass[i, 1]])
            if maxValue > highRawSigThresholdFactor:
                highRawInd.append(i)

        sixthPass = np.delete(fifthPass, highRawInd, 0)
        peakNormalizedPower = np.delete(peakNormalizedPower, highRawInd)
        peaktime = np.delete(peaktime, highRawInd)
        print(f"{len(sixthPass)} ripples kept after deleting unrealistic ripples")

        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

        ripples = {
            "times": sixthPass / SampFreq,
            "peaktime": peaktime / SampFreq,
            "peakPower": peakNormalizedPower,
            "Info": {"Date": dt_string},
            "DetectionParams": {
                "lowThres": self.lowthresholdFactor,
                "highThresh": self.highthresholdFactor,
                "ArtifactThresh": self.maxRipplePower,
                "lowFreq": lowFreq,
                "highFreq": highFreq,
                "minDuration": self.minRippleDuration,
                "maxDuration": self.maxRippleDuration,
            },
        }

        np.save(self._obj.sessinfo.files.ripple_evt, ripples)
        print(f"{self._obj.sessinfo.files.ripple_evt.name} created")

    def plot(self):
        """Gives a comprehensive view of the detection process with some statistics and examples
        """
        _, _, coords = self.best_chan_lfp()
        probemap = self._obj.recinfo.probemap()
        nChans = self._obj.recinfo.nChans
        changrp = self._obj.recinfo.channelgroups
        chosenShank = changrp[1] + changrp[2]
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
        gs = GridSpec(3, 10, figure=fig)
        fig.subplots_adjust(hspace=0.5)

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
        coords = np.asarray(coords)
        ax.plot(probemap[0], probemap[1], ".", color="#cdc6c6")
        ax.plot(coords[:, 0], coords[:, 1], "r.")
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

    def export2Neuroscope(self):
        times = self.time * 1000  # convert to ms
        file_neuroscope = self._obj.sessinfo.files.filePrefix.with_suffix(".evt.rpl")
        with file_neuroscope.open("w") as a:
            for beg, stop in times:
                a.write(f"{beg} start\n{stop} end\n")


class Spindle:
    # parameters in standard deviation
    lowthresholdFactor = 1.5
    highthresholdFactor = 4
    # parameters in ms
    minSpindleDuration = 350
    # maxSpindleDuration = 450
    mergeDistance = 125

    def __init__(self, obj):

        self._obj = obj

        if Path(self._obj.sessinfo.files.spindle_evt).is_file():

            spindle_evt = np.load(
                obj.sessinfo.files.spindle_evt, allow_pickle=True
            ).item()
            self.time = spindle_evt["times"]
            self.peakpower = spindle_evt["peakPower"]
            self.peaktime = spindle_evt["peaktime"]

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

        lfpinfo = np.load(self._obj.sessinfo.files.spindlelfp, allow_pickle=True).item()
        chan = np.asarray(lfpinfo["channel"])
        lfp = np.asarray(lfpinfo["lfp"])
        coords = lfpinfo["coords"]

        return lfp, chan, coords

    def channels(self, viewselection=1):
        """Channel which represent high spindle power during nrem across all channels

        """
        sampleRate = self._obj.recinfo.lfpSrate
        duration = 1 * 60 * 60  # chunk of lfp in seconds
        nyq = 0.5 * sampleRate  # Nyquist frequency
        nChans = self._obj.recinfo.nChans
        badchans = self._obj.recinfo.badchans
        allchans = self._obj.recinfo.channels
        changrp = self._obj.recinfo.channelgroups
        nShanks = self._obj.recinfo.nShanks
        probemap = self._obj.recinfo.probemap()
        brainChannels = [item for sublist in changrp[:nShanks] for item in sublist]
        dict_probemap = dict(zip(brainChannels, zip(probemap[0], probemap[1])))

        states = self._obj.brainstates.states
        states = states.loc[states["name"] == "nrem", ["start", "end"]]
        nremframes = (np.asarray(states.values.tolist()) * sampleRate).astype(int)
        reqframes = []
        for (start, end) in nremframes:
            reqframes.extend(np.arange(start, end))

        reqframes = np.asarray(reqframes)

        fileName = self._obj.sessinfo.recfiles.eegfile
        lfpAll = np.memmap(fileName, dtype="int16", mode="r")
        lfpAll = np.memmap.reshape(lfpAll, (int(len(lfpAll) / nChans), nChans))
        lfpCA1 = lfpAll[reqframes, :]

        # exclude badchannels
        lfpCA1 = np.delete(lfpCA1, badchans, 1)
        goodChans = np.setdiff1d(allchans, badchans, assume_unique=True)  # keeps order

        # filter and hilbet amplitude for each channel
        hilbertfast = lambda x: sg.hilbert(x, fftpack.next_fast_len(len(x)))[: len(x)]
        avgSpindle = np.zeros(lfpCA1.shape[1])
        for i in range(lfpCA1.shape[1]):
            spindleband = filt.filter_spindle(lfpCA1[:, i])
            amplitude_envelope = np.abs(hilbertfast(spindleband))
            avgSpindle[i] = np.mean(amplitude_envelope)

        spindleamp_chan = dict(zip(goodChans, avgSpindle))

        # plt.plot(probemap[0], probemap[1], ".", color="#bfc0c0")
        bestchan = goodChans[np.argmax(avgSpindle)]
        lfp = lfpAll[:, np.argmax(avgSpindle)]
        coords = dict_probemap[bestchan]

        # the reason metricAmp name was used to allow using other metrics such median
        bestspindlechan = dict(
            zip(["channel", "lfp", "coords"], [bestchan, lfp, coords],)
        )

        filename = self._obj.sessinfo.files.spindlelfp
        np.save(filename, bestspindlechan)

    def detect(self):
        """ripples lfp nchans x time

        Returns:
            [type] -- [description]
        """
        sampleRate = self._obj.recinfo.lfpSrate
        lowFreq = 9
        highFreq = 18

        spindlelfp, _, _ = self.best_chan_lfp()
        signal = np.asarray(spindlelfp, dtype=np.float)

        yf = filt.filter_spindle(signal, ax=-1)
        hilbertfast = lambda x: sg.hilbert(x, fftpack.next_fast_len(len(x)))[: len(x)]
        amplitude_envelope = np.abs(hilbertfast(yf))
        zscsignal = stats.zscore(amplitude_envelope, axis=-1)

        # delete ripples in noisy period
        deadfile = (self._obj.sessinfo.files.filePrefix).with_suffix(".dead")
        if deadfile.is_file():
            with deadfile.open("r") as f:
                noisy = []
                for line in f:
                    epc = line.split(" ")
                    epc = [float(_) for _ in epc]
                    noisy.append(epc)
                noisy = np.asarray(noisy)
                noisy = ((noisy / 1000) * sampleRate).astype(int)

            for noisy_ind in range(noisy.shape[0]):
                st = noisy[noisy_ind, 0]
                en = noisy[noisy_ind, 1]
                numnoisy = en - st
                zscsignal[st:en] = np.zeros((numnoisy))

        # hilbert transform --> binarize by > than lowthreshold
        ThreshSignal = np.diff(np.where(zscsignal > self.lowthresholdFactor, 1, 0))
        start_spindle = np.where(ThreshSignal == 1)[0]
        stop_spindle = np.where(ThreshSignal == -1)[0]

        if start_spindle[0] > stop_spindle[0]:
            stop_spindle = stop_spindle[1:]
        if start_spindle[-1] > stop_spindle[-1]:
            start_spindle = start_spindle[:-1]

        firstPass = np.vstack((start_spindle, stop_spindle)).T
        print(f"{len(firstPass)} spindles detected initially")

        # ===== merging close spindles =========
        minInterspindleSamples = self.mergeDistance / 1000 * sampleRate
        secondPass = []
        spindle = firstPass[0]
        for i in range(1, len(firstPass)):
            if firstPass[i, 0] - spindle[1] < minInterspindleSamples:
                # Merging spindles
                spindle = [spindle[0], firstPass[i, 1]]
            else:
                secondPass.append(spindle)
                spindle = firstPass[i]
        secondPass.append(spindle)
        secondPass = np.asarray(secondPass)
        print(f"{len(secondPass)} spindles remaining after merging")

        # =======delete spindles with less than threshold power
        thirdPass = []
        peakNormalizedPower, peaktime = [], []
        for i in range(0, len(secondPass)):
            maxValue = max(zscsignal[secondPass[i, 0] : secondPass[i, 1]])
            if maxValue >= self.highthresholdFactor:
                thirdPass.append(secondPass[i])
                peakNormalizedPower.append(maxValue)
                peaktime.append(
                    [
                        secondPass[i, 0]
                        + np.argmax(zscsignal[secondPass[i, 0] : secondPass[i, 1]])
                    ]
                )

        thirdPass = np.asarray(thirdPass)
        spindle_duration = np.diff(thirdPass, axis=1) / sampleRate * 1000
        print(f"{len(thirdPass)} spindles remaining after deleting weak spindles")

        # delete very short spindles
        shortspindles = np.where(spindle_duration < self.minSpindleDuration)[0]
        fourthPass = np.delete(thirdPass, shortspindles, 0)
        peakNormalizedPower = np.delete(peakNormalizedPower, shortspindles)
        spindle_duration = np.delete(spindle_duration, shortspindles)
        peaktime = np.delete(peaktime, shortspindles)
        print(f"{len(fourthPass)} spindles remaining after deleting short spindles")

        # delete spindles in non-nrem periods
        states = self._obj.brainstates.states
        states = states.loc[states["name"] == "nrem", ["start", "end"]]
        nremframes = (np.asarray(states.values.tolist()) * sampleRate).astype(int)
        reqframes = []
        for (start, end) in nremframes:
            reqframes.extend(np.arange(start, end))
        reqframes = np.asarray(reqframes)

        outside_spindles = []
        for ind, (start, _) in enumerate(fourthPass):
            if start not in reqframes:
                outside_spindles.extend([ind])
        outside_spindles = np.asarray(outside_spindles)

        fifthPass = np.delete(fourthPass, outside_spindles, 0)
        peakNormalizedPower = np.delete(peakNormalizedPower, outside_spindles)
        spindle_duration = np.delete(spindle_duration, outside_spindles)
        peaktime = np.delete(peaktime, outside_spindles)

        print(f"{len(fifthPass)} spindles finally kept after excluding outside nrem")

        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

        spindles = dict()
        spindles["times"] = fifthPass / sampleRate
        spindles["peakPower"] = peakNormalizedPower
        spindles["peaktime"] = peaktime / sampleRate
        spindles["DetectionParams"] = {
            "lowThres": self.lowthresholdFactor,
            "highThresh": self.highthresholdFactor,
            "lowFreq": lowFreq,
            "highFreq": highFreq,
            "minDuration": self.minSpindleDuration,
        }
        spindles["Info"] = {"Date": dt_string}

        # return spindles

        np.save(self._obj.sessinfo.files.spindle_evt, spindles)
        print(f"{self._obj.sessinfo.files.spindle_evt.name} created")

    def plot(self):
        """Gives a comprehensive view of the detection process with some statistics and examples
        """
        _, spindlechan, coord = self.best_chan_lfp()
        eegSrate = self._obj.recinfo.lfpSrate
        probemap = self._obj.recinfo.probemap()
        nChans = self._obj.recinfo.nChans
        changrp = self._obj.recinfo.channelgroups
        chosenShank = changrp[1] + changrp[2]
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
        frames = times * eegSrate
        nspindles = len(peakpower)

        fig = plt.figure(1, figsize=(6, 10))
        gs = GridSpec(3, 10, figure=fig)
        fig.subplots_adjust(hspace=0.5)

        spindles_to_plot = list(range(5)) + list(range(nspindles - 5, nspindles))
        for ind, spindle in enumerate(spindles_to_plot):
            print(spindle)
            start = int(frames[spindle, 0])
            end = int(frames[spindle, 1])
            lfp = stats.zscore(eegdata[start:end, :])
            ripplebandlfp = filt.filter_spindle(lfp, ax=0)
            # lfp = (lfp.T - np.median(lfp, axis=1)).T
            lfp = lfp + np.linspace(40, 0, lfp.shape[1])
            ripplebandlfp = ripplebandlfp + np.linspace(40, 0, lfp.shape[1])
            duration = (lfp.shape[0] / eegSrate) * 1000  # in ms

            ax = fig.add_subplot(gs[1, ind])
            ax.plot(lfp, "#fa761e", linewidth=0.8)
            ax.set_title(
                f"zsc = {round(peakpower[spindle],2)}, {round(duration)} ms", loc="left"
            )
            # ax.set_xlim([0, self.maxSpindleDuration / 1000 * eegSrate])
            ax.axis("off")

            ax = fig.add_subplot(gs[2, ind])
            ax.plot(ripplebandlfp, linewidth=0.8, color="#594f4f")
            # ax.set_title(f"{round(peakpower[ripple],2)}")
            # ax.set_xlim([0, self.maxSpindleDuration / 1000 * eegSrate])
            ax.axis("off")

        ax = fig.add_subplot(gs[0, 0])
        ax.text(
            0,
            0.8,
            f" highThresh ={self.highthresholdFactor}\n lowThresh ={self.lowthresholdFactor}\n minDuration = {self.minSpindleDuration}\n mergeSpindle = {self.mergeDistance} \n #Spindles = {len(peakpower)}",
        )
        ax.axis("off")

        ax = fig.add_subplot(gs[0, 1:4])
        coord = np.asarray(coord)
        ax.plot(probemap[0], probemap[1], ".", color="#cdc6c6")
        ax.plot(coord[0], coord[1], "r.")
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
        fig.suptitle(f"Spindle detection of {subname}")

    def export2Neuroscope(self):
        times = self.time * 1000  # convert to ms
        file_neuroscope = self._obj.sessinfo.files.filePrefix.with_suffix(".evt.spn")
        with file_neuroscope.open("w") as a:
            for beg, stop in times:
                a.write(f"{beg} start\n{stop} end\n")
