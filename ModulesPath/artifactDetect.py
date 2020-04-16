import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sg
import scipy.stats as stat
import pandas as pd
from numpy.fft import fft
import scipy.ndimage as filtSig
from pathlib import Path


class findartifact:

    lfpsRate = 1250
    art_thresh = 2

    def __init__(self, obj):
        self._obj = obj
        # self._myinfo = recinfo(basePath)

    def usingZscore(self):
        """
        calculating periods to exclude for analysis using simple z-score measure 
        """
        nChans = self._obj.recinfo.nChans
        Data = np.memmap(self._obj.sessinfo.recfiles.eegfile, dtype="int16", mode="r")
        Data1 = np.memmap.reshape(Data, (int(len(Data) / nChans), nChans))
        chanData = Data1[:, 17]

        zsc = np.abs(stat.zscore(chanData))

        artifact_binary = np.where(zsc > self.art_thresh, 1, 0)
        artifact_binary = np.concatenate(([0], artifact_binary, [0]))
        artifact_diff = np.diff(artifact_binary)

        artifact_start = np.where(artifact_diff == 1)[0]
        artifact_end = np.where(artifact_diff == -1)[0]

        firstPass = np.vstack((artifact_start - 10, artifact_end + 2)).T

        minInterArtifactDist = 5 * self.lfpsRate
        secondPass = []
        artifact = firstPass[0]
        for i in range(1, len(artifact_start)):
            if firstPass[i, 0] - artifact[1] < minInterArtifactDist:
                # Merging artifacts
                artifact = [artifact[0], firstPass[i, 1]]
            else:
                secondPass.append(artifact)
                artifact = firstPass[i]

        secondPass.append(artifact)

        # converting to required time units for various puposes------
        artifact_ms = np.asarray(secondPass) / (self.lfpsRate / 1000)  # ms
        artifact_s = np.asarray(secondPass) / self.lfpsRate  # seconds

        # writing to file for visualizing in neuroscope and spyking circus
        start_neuroscope = self._obj.sessinfo.files.filePrefix.with_suffix(".evt.sta")
        end_neuroscope = self._obj.sessinfo.files.filePrefix.with_suffix(".evt.end")
        circus_file = self._obj.sessinfo.files.filePrefix.with_suffix(".dead")
        with start_neuroscope.open("w") as a, end_neuroscope.open(
            "w"
        ) as b, circus_file.open("w") as c:
            for beg, stop in artifact_ms:
                a.write(f"{beg} start\n")
                b.write(f"{stop} end\n")
                c.write(f"{beg} {stop}\n")

        return zsc
        # self.artifact_time = artifact_s

    def usingCorrelation(self):

        Data = np.memmap(self._recfiles.eegfile, dtype="int16", mode="r")
        Data1 = np.memmap.reshape(
            Data, (int(len(Data) / self._myinfo.nChans), self._myinfo.nChans)
        )
        chan1 = Data1[:, 17]
        chan2 = Data1[:, 18]

        corr = []
        for i in range(0, len(chan1) - 1250, 1250):

            x = chan1[i : i + 1250]
            y = chan2[i : i + 1250]

            corr.append(np.correlate(x, y))

        return corr

    @property
    def plot(self):

        self.zsc_signal = zsc
        self.artifact_time = artifact_s

        plt.plot(self.zsc_signal)

    def createCleanDat(self):

        for shankID in range(3, 9):
            print(shankID)

            DatFileOG = (
                folderPath
                + "Shank"
                + str(shankID)
                + "/RatJDay2_Shank"
                + str(shankID)
                + ".dat"
            )
            DestFolder = (
                folderPath
                + "Shank"
                + str(shankID)
                + "/RatJDay2_Shank"
                + str(shankID)
                + "_denoised.dat"
            )

            nChans = 8
            SampFreq = 30000

            b = []
            for i in range(len(Data_start)):

                start_time = Data_start[i]
                end_time = Data_end[i]

                duration = end_time - start_time  # in seconds
                b.append(
                    np.memmap(
                        DatFileOG,
                        dtype="int16",
                        mode="r",
                        offset=2 * nChans * int(SampFreq * start_time),
                        shape=(nChans * int(SampFreq * duration)),
                    )
                )

            c = np.memmap(
                DestFolder, dtype="int16", mode="w+", shape=sum([len(x) for x in b])
            )

            del c
            d = np.memmap(
                DestFolder, dtype="int16", mode="r+", shape=sum([len(x) for x in b])
            )

            sizeb = [0]
            sizeb.extend([len(x) for x in b])
            sizeb = np.cumsum(sizeb)

            for i in range(len(b)):

                d[sizeb[i] : sizeb[i + 1]] = b[i]
                # d[len(b[i]) : len(b1) + len(b2)] = b2
            del d
            del b
