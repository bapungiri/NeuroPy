#%%====== Import statements==========
import numpy as np
import matplotlib.pyplot as plt
from SpectralAnalysis import bestRippleChannel, bestThetaChannel, lfpSpectrogram
from lfpDetect import swr
import os
from datetime import datetime as dt

import matplotlib.style
import matplotlib as mpl

mpl.style.use("default")

# %load_ext autoreload
# %autoreload 2


#  ========== Ripple Detection ============


class RippleDetect:
    sRate = 1250

    def __init__(self, basePath):
        self.sessionName = basePath.split("/")[-2] + basePath.split("/")[-1]
        self.basePath = basePath
        for file in os.listdir(basePath):
            if file.endswith(".eeg"):
                self.subname = file[:-4]
                self.filename = os.path.join(basePath, file)
                self.filePrefix = os.path.join(basePath, file[:-4])

    def findRipples(self):
        epoch_time = np.load(self.filePrefix + "_epochs.npy", allow_pickle=True)
        recording_dur = epoch_time.item().get("POST")[1]  # in seconds
        pre = epoch_time.item().get("PRE")  # in seconds
        maze = epoch_time.item().get("MAZE")  # in seconds
        post = epoch_time.item().get("POST")  # in seconds
        self.basics = np.load(self.filePrefix + "_basics.npy", allow_pickle=True)
        self.nChans = self.basics.item().get("nChans")

        if not os.path.exists(self.filePrefix + "_BestRippleChans.npy"):

            badChannels = np.load(self.filePrefix + "_badChans.npy")
            self.bestRippleChannels = bestRippleChannel(
                self.basePath,
                sampleRate=self.sRate,
                nChans=self.nChans,
                badChannels=badChannels,
                saveRippleChan=1,
            )

            swr(self.basePath, sRate=self.sRate, PlotRippleStat=0, savefile=1)

        self.ripples = np.load(self.filePrefix + "_ripples.npy", allow_pickle=True)

        self.ripplesTime = self.ripples.item().get("timestamps")
        self.rippleStart = self.ripplesTime[:, 0]
        self.histRipple, self.edges = np.histogram(self.rippleStart, bins=20)
        self.edges = self.edges / (1250 * 3600)

    # def lfpSpect(self):
    #     self.Pxx, self.freq, self.time, self.sampleData = lfpSpectrogram(
    #         self.basePath, self.sRate, nChans=self.nChans, loadfrom=1
    #     )
    #     f_req_ind = np.where(self.freq < 50)[0]

    #     self.f_req = self.freq[f_req_ind]
    #     self.Pxx_req = self.Pxx[f_req_ind, :]
    #     self.Pxx_req = np.flipud(self.Pxx_req)
    #     self.time = self.time / 3600

    def sessionInfo(self):
        self.Date = self.ripples["DetectionParams"]


basePath = [
    "/data/Clustering/SleepDeprivation/RatJ/Day1/",
    "/data/Clustering/SleepDeprivation/RatJ/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day1/",
    # "/data/Clustering/SleepDeprivation/RatK/Day2/",
    "/data/Clustering/SleepDeprivation/RatN/Day1/",
    "/data/Clustering/SleepDeprivation/RatN/Day2/",
]

Ripple_inst = [RippleDetect(x) for x in basePath]

for i, sess in enumerate(Ripple_inst):
    sess.findRipples()
    # Ripple_inst[i].findRipples()


plt.clf()
for i, sess in enumerate(Ripple_inst):
    plt.plot(sess.histRipple)


# fig = plt.figure(1)
# for i in range(4):
#     # plt.subplot(4, 1, i + 1)
#     ax1 = fig.add_subplot(4, 1, i + 1)

#     ax1.imshow(
#         np.log(Ripple_inst[i].Pxx_req),
#         cmap="OrRd",
#         aspect="auto",
#         vmax=12,
#         vmin=4,
#         extent=[
#             Ripple_inst[i].time[0],
#             Ripple_inst[i].time[-1],
#             Ripple_inst[i].f_req[0],
#             Ripple_inst[i].f_req[-1],
#         ],
#     )
#     ax2 = ax1.twinx()
#     ax2.plot(Ripple_inst[i].edges[:-1], Ripple_inst[i].histRipple, "k")
#     plt.xlim([0, Ripple_inst[i].time[-1]])
#     plt.title(Ripple_inst[i].sessionName)

