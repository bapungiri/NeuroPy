import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path
from dataclasses import dataclass


class spikes:
    def __init__(self, obj):
        self._obj = obj

        filename = self._obj.sessinfo.files.spikes
        if filename.is_file():
            spikes = np.load(filename, allow_pickle=True).item()
            self.times = spikes["times"]
            self.info = spikes["info"].reset_index()

        self.stability = Stability(obj)
        self.dynamics = firingDynamics(obj)

    def removeDoubleSpikes(self):
        nShanks = self._obj.recinfo.nShanks
        name = self._obj.sessinfo.session.name
        day = self._obj.sessinfo.session.day
        sRate = self._obj.recinfo.sampfreq
        basePath = self._obj.sessinfo.basePath
        clubasePath = Path(basePath, "/spykcirc")
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

    def fromCircus(self, fileformat="diff_folder"):

        if fileformat == "diff_folder":
            nShanks = self._obj.recinfo.nShanks
            sRate = self._obj.recinfo.sampfreq
            name = self._obj.sessinfo.session.name
            day = self._obj.sessinfo.session.day
            basePath = self._obj.sessinfo.basePath
            clubasePath = Path(basePath, "spykcirc")
            spkall, info, shankID = [], [], []
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
                goodCellsID = cluinfo.id[cluinfo["q"] < 10].tolist()
                info.append(cluinfo.loc[cluinfo["q"] < 10])
                shankID.extend(shank * np.ones(len(goodCellsID)))

                for i in range(len(goodCellsID)):
                    clu_spike_location = spktime[np.where(cluID == goodCellsID[i])[0]]
                    spkall.append(clu_spike_location / sRate)

            spkinfo = pd.concat(info, ignore_index=True)
            spkinfo["shank"] = shankID
            spktimes = spkall

        if fileformat == "same_folder":
            nShanks = self._obj.recinfo.nShanks
            sRate = self._obj.recinfo.sampfreq
            subname = self._obj.sessinfo.session.subname
            basePath = self._obj.sessinfo.basePath
            changroup = self._obj.recinfo.channelgroups
            clubasePath = Path(basePath, "spykcirc")

            clufolder = Path(clubasePath, subname, subname + ".GUI",)
            spktime = np.load(clufolder / "spike_times.npy")
            cluID = np.load(clufolder / "spike_clusters.npy")
            cluinfo = pd.read_csv(clufolder / "cluster_info.tsv", delimiter="\t")
            goodCellsID = cluinfo.id[cluinfo["q"] < 10].tolist()
            info = cluinfo.loc[cluinfo["q"] < 10]
            peakchan = info["ch"]
            shankID = [
                sh + 1
                for chan in peakchan
                for sh, grp in enumerate(changroup)
                if chan in grp
            ]

            spkall = []
            for i in range(len(goodCellsID)):
                clu_spike_location = spktime[np.where(cluID == goodCellsID[i])[0]]
                spkall.append(clu_spike_location / sRate)

            info["shank"] = shankID
            spkinfo = info
            spktimes = spkall
            # self.shankID = np.asarray(shankID)

        spikes_ = {"times": spktimes, "info": spkinfo}
        filename = self._obj.sessinfo.files.spikes
        np.save(filename, spikes_)

    def fromNeurosuite(self):
        pass

    def fromKilosort2(self):
        pass


class Stability:
    def __init__(self, obj):
        self._obj = obj
        filePrefix = self._obj.sessinfo.files.filePrefix

        @dataclass
        class files:
            stability: str = Path(str(filePrefix) + "_stability.npy")

        self.files = files()

        if self.files.stability.is_file():
            self._load()

    def _load(self):
        data = np.load(self.files.stability, allow_pickle=True).item()
        self.info = data["stableinfo"]
        self.isStable = data["isStable"]
        self.bins = data["bins"]
        self.thresh = data["thresh"]

    def firingRate(self, bins=None, thresh=0.3):

        spks = self._obj.spikes.times
        nCells = len(spks)

        # ---- goes to default mode of PRE-POST stability --------
        if bins is None:
            pre = self._obj.epochs.pre
            pre = self._obj.utils.getinterval(period=pre, nbins=3)

            post = self._obj.epochs.post
            post = self._obj.utils.getinterval(period=post, nbins=5)
            total_dur = self._obj.epochs.totalduration
            mean_frate = self._obj.spikes.info.fr
            bins = pre + post
            nbins = len(bins)

        # --- number of spikes in each bin ------
        bin_dur = np.asarray([np.diff(window) for window in bins]).squeeze()
        total_dur = np.sum(bin_dur)
        nspks_bin = np.asarray(
            [np.histogram(cell, bins=np.concatenate(bins))[0][::2] for cell in spks]
        )
        assert nspks_bin.shape[0] == nCells

        total_spks = np.sum(nspks_bin, axis=1)

        if bins is not None:
            nbins = len(bins)
            mean_frate = total_spks / total_dur

        # --- calculate meanfr in each bin and the fraction of meanfr over all bins
        frate_bin = nspks_bin / np.tile(bin_dur, (nCells, 1))
        fraction = frate_bin / mean_frate.reshape(-1, 1)
        assert frate_bin.shape == fraction.shape

        isStable = np.where(fraction >= thresh, 1, 0)
        spkinfo = self._obj.spikes.info[["q", "shank"]].copy()
        spkinfo["stable"] = isStable.all(axis=1).astype(int)

        stbl = {
            "stableinfo": spkinfo,
            "isStable": isStable,
            "bins": bins,
            "thresh": thresh,
        }
        np.save(self.files.stability, stbl)
        self._load()

    def refPeriodViolation(self):

        spks = self._obj.spikes.times

        fp = 0.05  # accepted contamination level
        T = self._obj.epochs.totalduration
        taur = 2e-3
        tauc = 1e-3
        nbadspikes = lambda N: 2 * (taur - tauc) * (N ** 2) * (1 - fp) * fp / T

        nSpks = [len(_) for _ in spks]
        expected_violations = [nbadspikes(_) for _ in nSpks]

        self.expected_violations = np.asarray(expected_violations)

        isi = [np.diff(_) for _ in spks]
        ref = np.array([0, 0.002])
        zerolag_spks = [np.histogram(_, bins=ref)[0] for _ in isi]

        self.violations = np.asarray(zerolag_spks)

    def isolationDistance(self):
        pass


class firingDynamics:
    def __init__(self, obj):
        self._obj = obj

    def fRate(self):
        pass

    def plotfrate(self):
        pass

    def plotRaster(self):
        pass
