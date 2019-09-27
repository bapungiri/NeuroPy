#%%====== Import statements==========
import numpy as np
import matplotlib.pyplot as plt
from SpectralAnalysis import bestRippleChannel, bestThetaChannel, lfpSpectrogram
from lfpDetect import swr
import os
from datetime import datetime as dt


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
        self.Starttime = dt.strptime(self.sessionnName[-19:], "%Y-%m-%d_%H-%M-%S")
        self.ripples = swr(self.basePath, sRate=self.sRate, PlotRippleStat=1)
        self.ripplesTime = self.ripples["timestamps"]
        self.rippleStart = self.ripplesTime[:, 0]
        self.histRipple, self.edges = np.histogram(self.rippleStart, bins=20)

    def lfpSpect(self):
        self.spect, self.freq, self.time, self.sampleData = lfpSpectrogram(
            self.basePath, self.sRate, nChans=self.nChans, loadfrom=1
        )
        self.time = self.time / 3600
        self.spect = np.flip(self.spect)

    def sessionInfo(self):
        self.Date = self.ripples["DetectionParams"]


folderPath = [
    "/home/bapung/Documents/ClusteringHub/EEGAnlaysis/RatJ/RatJ_2019-05-31_03-55-36/",
    "/home/bapung/Documents/ClusteringHub/EEGAnlaysis/RatJ/RatJ_2019-06-02_03-59-19/",
    "/home/bapung/Documents/ClusteringHub/EEGAnlaysis/RatK/RatK_2019-08-06_03-44-01/",
    "/home/bapung/Documents/ClusteringHub/EEGAnlaysis/RatK/RatK_2019-08-08_04-00-00/",
]


# dict_sessions = {
#     "RatJ_SD": RippleDetect(folderPath[0]),
#     "RatJ_NoSD": RippleDetect(folderPath[1]),
#     "RatK_SD": RippleDetect(folderPath[2]),
#     "RatK_NoSD": RippleDetect(folderPath[3]),
# }
badChannels_all = [
    [1, 3, 7] + list(range(65, 76)),
    [1, 3, 7, 6, 65, 66, 67],
    np.arange(65, 135),
    [102, 106, 127, 128],
]

nChans_all = [75, 67, 134, 134]

Ripple_inst = [RippleDetect(folderPath[i]) for i in range(4)]

for i in range(4):
    Ripple_inst[i].badChannels = badChannels_all[i]
    Ripple_inst[i].nChans = nChans_all[i]


sessions = ["RatJ_SD", "RatJ_NoSD", "RatK_SD", "RatK_NoSD"]
spect_sessions = [RatJ_NoSD.spect]

plt.clf()
for i in range(1):
    plt.subplot(4, 1, i + 1)
    plt.imshow(
        RatJ_NoSD.spect[10:200, :],
        cmap="YlGn",
        vmax=0.01,
        extent=[
            np.min(RatJ_NoSD.time),
            np.max(RatJ_NoSD.time),
            np.min(RatJ_NoSD.freq),
            np.max(RatJ_NoSD.freq),
        ],
        aspect="auto",
    )


# %%
