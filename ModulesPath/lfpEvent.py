import numpy as np
import matplotlib.pyplot as plt
from parsePath import name2path
import scipy.signal as sg
import scipy.stats as stat
import pandas as pd
import scipy.fftpack as ft
import scipy.ndimage as smth
import numpy.random as rnd
import os
from datetime import datetime


class fromLfp(name2path):

    lfpsRate = 1250
    binSize = 0.250  # in seconds

    def __init__(self, basePath):
        super().__init__(basePath)

    ##======= hippocampal slow wave detection =============
    def hswa(self):
        """
        fileName: name of the .eeg file
        sampleRate: sampling frequency of eeg
        """

        nyq = 0.5 * self.lfpsRate
        lowdelta = 0.5  # in Hz
        highdelta = 4  # in Hz
        deltachan = np.load(
            str(self.filePrefix) + "_BestRippleChans.npy", allow_pickle=True
        ).item()
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

    def swr(self, PlotRippleStat=0):

        SampFreq = self.lfpsRate
        nyq = 0.5 * SampFreq
        lowFreq = 150
        highFreq = 240
        lowthresholdFactor = 1
        highThresholdFactor = 2
        # TODO chnage raw amplitude threshold to something statistical
        highRawSigThresholdFactor = 15000
        minRippleDuration = 20  # in milliseconds
        maxRippleDuration = 800  # in milliseconds
        maxRipplePower = 60  # in normalized power units

        fileName = str(self.filePrefix) + "_BestRippleChans.npy"
        lfpCA1 = np.load(
            str(self.filePrefix) + "_BestRippleChans.npy", allow_pickle=True
        )

        signal = lfpCA1.item()
        signal = signal["BestChan"]
        signal = np.array(signal, dtype=np.float)  # convert data to float
        zscoreRawSig = stat.zscore(signal)

        b, a = sg.butter(3, [lowFreq / nyq, highFreq / nyq], btype="bandpass")
        yf = sg.filtfilt(b, a, signal)

        squared_signal = np.square(yf)
        normsquaredsignal = stat.zscore(squared_signal)

        # getting an envelope of the signal
        # analytic_signal = sg.hilbert(yf)
        # amplitude_envelope = stat.zscore(np.abs(analytic_signal))

        windowLength = SampFreq / SampFreq * 11
        window = np.ones((int(windowLength),)) / windowLength

        smoothSignal = sg.filtfilt(window, 1, squared_signal, axis=0)
        zscoreSignal = stat.zscore(smoothSignal)

        hist_zscoresignal, edges_zscoresignal = np.histogram(
            zscoreSignal, bins=np.linspace(0, 6, 100)
        )

        ThreshSignal = np.diff(np.where(zscoreSignal > lowthresholdFactor, 1, 0))
        start_ripple = np.argwhere(ThreshSignal == 1)
        stop_ripple = np.argwhere(ThreshSignal == -1)

        # print(start_ripple.shape, stop_ripple.shape)
        firstPass = np.concatenate((start_ripple, stop_ripple), axis=1)

        # TODO delete half ripples in begining or end

        # ===== merging close ripples
        minInterRippleSamples = 30 / 1000 * SampFreq
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
            if maxValue > highThresholdFactor:
                thirdPass.append(secondPass[i])
                peakNormalizedPower.append(maxValue)

        thirdPass = np.asarray(thirdPass)

        ripple_duration = np.diff(thirdPass, axis=1) / SampFreq * 1000

        # delete very short ripples
        shortRipples = np.where(ripple_duration < minRippleDuration)[0]
        thirdPass = np.delete(thirdPass, shortRipples, 0)
        peakNormalizedPower = np.delete(peakNormalizedPower, shortRipples)

        # delete ripples with unrealistic high power
        artifactRipples = np.where(peakNormalizedPower > maxRipplePower)[0]
        fourthPass = np.delete(thirdPass, artifactRipples, 0)
        peakNormalizedPower = np.delete(peakNormalizedPower, artifactRipples)

        # delete very long ripples
        veryLongRipples = np.where(ripple_duration > maxRippleDuration)[0]
        fifthPass = np.delete(fourthPass, veryLongRipples, 0)
        peakNormalizedPower = np.delete(peakNormalizedPower, veryLongRipples)

        # delete ripples which have unusually high amp in raw signal (takes care of disconnection)
        highRawInd = []
        for i in range(0, len(fifthPass)):
            maxValue = max(signal[fifthPass[i, 0] : fifthPass[i, 1]])
            if maxValue > highRawSigThresholdFactor:
                highRawInd.append(i)

        sixthPass = np.delete(fifthPass, highRawInd, 0)
        peakNormalizedPower = np.delete(peakNormalizedPower, highRawInd)

        print(sixthPass.shape)
        # TODO delete sharp ripple like artifacts
        # Maybe unrelistic high power has taken care of this but check to confirm

        # selecting some example ripples
        idx = rnd.randint(0, sixthPass.shape[0], 5, dtype="int")
        example_ripples = []
        example_ripples_duration = []  # in frames
        for i in range(5):
            example_ripples.append(
                signal[sixthPass[idx[i], 0] - 125 : sixthPass[idx[i], 1] + 125]
            )
            example_ripples_duration.append(sixthPass[idx[i], 1] - sixthPass[idx[i], 0])

        # selecting high power ripples
        highpoweredRipples = np.argsort(peakNormalizedPower)
        for i in range(5):
            example_ripples.append(
                signal[sixthPass[idx[i], 0] - 125 : sixthPass[idx[i], 1] + 125]
            )
            example_ripples_duration.append(sixthPass[idx[i], 1] - sixthPass[idx[i], 0])

        # ====== Plotting some results===============
        if PlotRippleStat == 1:
            flat_ripples = [item for sublist in example_ripples for item in sublist]
            ripple_duration_hist, duration_edges = np.histogram(
                example_ripples_duration, bins=20
            )

            numRow, numCol = 3, 2
            plt.figure()
            plt.subplot(numRow, 1, 1)
            plt.plot(flat_ripples)

            plt.subplot(numRow, 2, 3)
            # plt.plot(duration_edges, ripple_duration_hist)
            sns.distplot(ripple_duration, bins=None)
            plt.xlabel("Ripple duration (ms)")
            plt.ylabel("Counts")
            plt.title("Ripple duration distribution")

            plt.subplot(numRow, 2, 4)
            # sns.set_style("darkgrid")
            sns.distplot(peakNormalizedPower, bins=np.arange(1, 100))
            plt.xlabel("Normalized Power")
            plt.ylabel("Counts")
            plt.title("Ripple Power Distribution")

        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

        ripples = dict()
        ripples["timestamps"] = sixthPass
        ripples["DetectionParams"] = {
            "lowThres": lowthresholdFactor,
            "highThresh": highThresholdFactor,
            "ArtifactThresh": maxRipplePower,
            "lowFreq": lowFreq,
            "highFreq": highFreq,
            "samplingRate": SampFreq,
            "minDuration": minRippleDuration,
            "maxDuration": maxRippleDuration,
        }
        ripples["Info"] = {"Date": dt_string, "DetectorName": "lfpDetect/swr"}

        np.save(str(self.filePrefix) + "_ripples.npy", ripples)

        return ripples
