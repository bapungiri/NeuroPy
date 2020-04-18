import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path


class spikes:
    def __init__(self, obj):
        self._obj = obj
        self.stability = stability(obj)

    def extract(self):
        self._Circus()

    def removeDoubleSpikes(self):
        nShanks = self._obj.recinfo.nShanks
        name = self._obj.sessinfo.session.name
        day = self._obj.sessinfo.session.day
        sRate = self._obj.recinfo.sampfreq
        clubasePath = Path("/home/bapung/Documents/ClusteringHub/spykcirc", name, day)
        spkall, info, templates = [], []
        for shank in range(1, nShanks + 1):

            clufolder = Path(
                clubasePath,
                name + day + "Shank" + str(shank),
                name + day + "Shank" + str(shank) + ".GUI",
            )

            spktime = np.load(clufolder / "spike_times.npy")
            cluID = np.load(clufolder / "spike_clusters.npy")
            templates = np.loadc(clufolder / "amplitudes.npy")

        self.info = pd.concat(info)
        self.spks = spkall

    def _Circus(self):
        nShanks = self._obj.recinfo.nShanks
        name = self._obj.sessinfo.session.name
        day = self._obj.sessinfo.session.day
        sRate = self._obj.recinfo.sampfreq
        clubasePath = Path("/home/bapung/Documents/ClusteringHub/spykcirc", name, day)
        spkall, info = [], []
        for shank in range(1, nShanks + 1):

            clufolder = Path(
                clubasePath,
                name + day + "Shank" + str(shank),
                name + day + "Shank" + str(shank) + ".GUI",
            )

            # datFile = np.memmap(file + "Shank" + str(i) + ".dat", dtype="int16")
            # datFiledur = len(datFile) / (16 * sRate)
            spktime = np.load(clufolder / "spike_times.npy")
            cluID = np.load(clufolder / "spike_clusters.npy")
            cluinfo = pd.read_csv(clufolder / "cluster_info.tsv", delimiter="\t")
            goodCellsID = cluinfo.id[cluinfo["group"] == "good"].tolist()
            info.append(cluinfo.loc[cluinfo["group"] == "good"])

            for i in range(len(goodCellsID)):
                clu_spike_location = spktime[np.where(cluID == goodCellsID[i])[0]]
                spkall.append(clu_spike_location / sRate)

        self.info = pd.concat(info)
        self.spks = spkall

    def _Neurosuite(self):
        pass

    def _Kilosort2(self):
        pass


class stability:
    def __init__(self, obj):
        self._obj = obj

    def firingRate(self):
        pass

    def isolationDistance(self):
        pass
