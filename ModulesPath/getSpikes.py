import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path


class spikes:
    def __init__(self, obj):
        self._obj = obj

        filename = self._obj.sessinfo.files.spikes
        if filename.is_file():
            spikes = np.load(filename, allow_pickle=True).item()
            self.times = spikes["times"]
            self.info = spikes["info"]

        self.stability = stability(obj)
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

            spkinfo = pd.concat(info)
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


class stability:
    def __init__(self, obj):
        self._obj = obj

    def firingRate(self):
        pre = self._obj.epochs.pre
        maze = self._obj.epochs.maze
        post = self._obj.epochs.post
        total_dur = self._obj.epochs.totalduration

        pre_bin = np.linspace(pre[0], pre[1], 3)
        maze_bin = maze
        post_bin = np.linspace(post[0], post[1], 4)

        spks = self._obj.spikes.times

        bin_all = np.concatenate((pre, maze, post))
        total_spks = np.array(
            [np.histogram(x, bins=bin_all)[0][::2].sum() for x in spks]
        )
        mean_frate = total_spks / total_dur

        pre_spikecount = np.array([np.histogram(x, bins=pre_bin)[0] for x in spks])
        maze_spikecount = np.array([np.histogram(x, bins=maze_bin)[0] for x in spks])
        post_spikecount = np.array([np.histogram(x, bins=post_bin)[0] for x in spks])

        frate_pre = pre_spikecount / (np.mean(np.diff(pre_bin)))
        frate_maze = maze_spikecount / np.diff(maze)
        frate_post = post_spikecount / (np.mean(np.diff(post_bin)))

        frate = np.concatenate((frate_pre, frate_post), axis=1)

        fraction = frate / mean_frate.reshape(-1, 1)

        self.isStable = np.where(fraction > 0.3, 1, 0)
        self.unstable = np.unique(np.argwhere(fraction < 0.3)[:, 0])
        self.stable = np.setdiff1d(range(len(spks)), self.unstable)
        self.mean_frate = mean_frate

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
