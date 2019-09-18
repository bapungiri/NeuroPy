#%%====== Import statements==========
import numpy as np
import matplotlib.pyplot as plt
from SpectralAnalysis import bestRippleChannel, bestThetaChannel, lfpSpectrogram
from lfpDetect import swr
import os


# %load_ext autoreload
# %autoreload 2


# %% ======== Theta detection testuing==============
# basePath = (
#     "/home/bapung/Documents/ClusteringHub/EEGAnlaysis/RatK/RatK_2019-08-06_03-44-01/"
# )
# nChans = 134
# # subject = ''
# # fileName = basePath + subject + '/' + subject + '.eeg'
# subname = os.path.basename(os.path.normpath(basePath))
# fileName = basePath + subname + ".eeg"
# reqChan = 33
# b1 = np.memmap(fileName, dtype="int16", mode="r")
# ThetaExtract = b1[reqChan::nChans]

# np.save(basePath + subname + "_BestThetaChan.npy", ThetaExtract)


# subname = os.path.basename(os.path.normpath(basePath))
# bestThetaCheck = basePath + subname + "_BestThetaChan.npy"
# thetasec = np.load(bestThetaCheck)

# T = 1 / 1250
# N = len(thetasec)
# fftTheta = np.fft.fft(thetasec)
# xf = np.linspace(0.0, 1.0 / (2.0 * T), N / 2)

# plt.clf()
# plt.subplot(2, 1, 1)
# plt.plot(xf, 2.0 / N * np.abs(fftTheta[: N // 2]))

# plt.subplot(2, 1, 2)
# plt.plot(thetasec[0 : 5 * 1250])


# fftTheta = np.fft.fft()

# sampleRate = 1250
# nChans = 134
# badChannels = np.arange(65, 134)

# bestTheta = bestThetaChannel(
#     basePath, sampleRate, nChans, badChannels, saveThetaChan=1)

# sxx, f, t, sample = lfpSpectrogram(
#     basePath, sampleRate, nChans, 64, loadfrom=0)
# sxx = sxx[0:500, :]
# sxx = np.flip(sxx)

# plt.clf()
# plt.imshow(
#     sxx, extent=[0, 10, f[0], f[500]], aspect="auto", vmax=70000, cmap="YlGn")
# plt.ylabel("Frequency (Hz)")
# plt.xlabel("Time (h)")

# bestChannel = bestRippleChannel(basePath, sampleRate, nChans, badChannels)
# ripples = swr(basePath, bestChannel[0], sampleRate, nChans)

# rpple_times = ripples[0]
# ripple_start = rpple_times[:, 0]

# ripple_counts, bin_edges = np.histogram(ripple_start, bins=14)

# %% ========== Ripple Detection ============


class RippleDetect:
    nChans = 134
    sRate = 1250
    badChannels = np.arange(65, 134)

    def __init__(self, basePath):
        self.sessionnName = os.path.basename(os.path.normpath(basePath))
        self.basePath = basePath

    def findRipples(self):
        if not os.path.exists(
            self.basePath + self.sessionnName + "_BestRippleChans.npy"
        ):
            self.bestRippleChannels = bestRippleChannel(
                self.basePath,
                sampleRate=self.sRate,
                nChans=self.nChans,
                badChannels=self.badChannels,
                saveRippleChan=1,
            )

        self.ripples = swr(self.basePath, sRate=self.sRate, PlotRippleStat=1)
        self.ripplesTime = self.ripples["timestamps"]
        self.rippleStart = self.ripplesTime[:, 0]
        self.histRipple, self.edges = np.histogram(self.rippleStart, bins=20)

    def sessionInfo(self):
        self.Date = self.ripples["DetectionParams"]


folderPath = [
    "/home/bapung/Documents/ClusteringHub/EEGAnlaysis/RatJ/RatJ_2019-05-31_03-55-36/",
    "/home/bapung/Documents/ClusteringHub/EEGAnlaysis/RatJ/RatJ_2019-06-02_03-59-19/",
    "/home/bapung/Documents/ClusteringHub/EEGAnlaysis/RatK/RatK_2019-08-06_03-44-01/",
    "/home/bapung/Documents/ClusteringHub/EEGAnlaysis/RatK/RatK_2019-08-08_04-00-00/",
]


RatJ_SD = RippleDetect(folderPath[0])
RatJ_NoSD = RippleDetect(folderPath[1])
RatK_SD = RippleDetect(folderPath[2])
RatK_NoSD = RippleDetect(folderPath[3])

RatJ_NoSD.badChannels = [1, 3, 7, 6, 65, 66, 67]
RatJ_NoSD.nChans = 67
RatJ_NoSD.findRipples()

# RatJ_SD.badChannels = [1, 3, 7] + list(range(65, 76))
# RatJ_SD.nChans = 75
# RatJ_SD.findRipples()


# RippleDetect.badChannels = np.arange(65, 134)
# RatK_SleepDep = RippleDetect(basePath3)
# RatK_NoSleepDep = RippleDetect(basePath4)


# plt.clf()
# plt.plot(RatJ_SleepDep.histRipple, "r")
# plt.plot(RatJ_NoSleepDep.histRipple)
# plt.plot(RatK_SleepDep.histRipple, "k")
# plt.plot(RatK_NoSleepDep.histRipple)


# %%
