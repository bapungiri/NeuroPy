import numpy as np
import matplotlib.pyplot as plt
from parsePath import name2path
import scipy.signal as sg
import scipy.stats as stat
import pandas as pd
from numpy.fft import fft
import scipy.ndimage as filtSig
from pathlib import Path


class findartifact(name2path):

    lfpsRate = 1250
    art_thresh = 1.5

    def __init__(self, basePath):
        super().__init__(basePath)

    def gen_artifact_epoch(self):
        """
        calculating periods to exclude for analysis using simple z-score measure 
        """

        # TODO select channel using theta chan
        # thetachan = np.load(
        #     str(self.filePrefix) + "_BestThetaChan.npy", allow_pickle=True
        # ).item()
        Data = np.memmap(str(self.eegfile), dtype="int16", mode="r")
        Data1 = np.memmap.reshape(Data, (int(len(Data) / 67), 67))
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
        start_neuroscope = self.filePrefix.with_suffix(".evt.sta")
        end_neuroscope = self.filePrefix.with_suffix(".evt.end")
        circus_file = self.filePrefix.with_suffix(".dead")
        with start_neuroscope.open("w") as a, end_neuroscope.open(
            "w"
        ) as b, circus_file.open("w") as c:
            for beg, stop in artifact_ms:
                a.write(f"{beg} start\n")
                b.write(f"{stop} end\n")
                c.write(f"{beg} {stop}\n")

        self.zsc_signal = zsc
        self.artifact_time = artifact_s

