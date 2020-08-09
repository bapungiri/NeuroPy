import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from mathutil import parcorr_mult, getICA_Assembly
import scipy.stats as stats
import matplotlib.gridspec as gridspec


class Replay:
    def __init__(self, obj):
        self.expvar = ExplainedVariance(obj)
        self.bayesian = Bayesian(obj)
        self.assembly = CellAssembly(obj)


class Bayesian:
    def __init__(self, obj):
        self._obj = obj

    def correlation(self, template_time=None, match_time=None, cells=None):
        """Pairwise correlation between template window and matching windows

        Args:
            template_time (array_like, optional): in seconds
            match_time (array_like, optional): in seconds
            cells ([type], optional): cells to calculate the correlation for.

        Returns:
            [array]: returns correlation
        """

        if template_time is None:
            template_time = self._obj.epochs.maze

        if match_time is None:
            match_time = self._obj.epochs.post

        if cells is None:
            unstable_units = self._obj.spikes.stability.unstable
            # stable_units = self._obj.spikes.stability.stable
            # stable_units = list(range(len(spks)))
            stable_units = self._obj.spikes.stability.stable
            quality = np.asarray(self._obj.spikes.info.q)
            pyr = np.where(quality < 5)[0]
            stable_pyr = stable_units[np.isin(stable_units, pyr)]

        spks = self._obj.spikes.times
        print(stable_pyr)

        # spks = [spks[x] for x in stable_units]

        nUnits = len(spks)
        windowSize = self.timeWindow
        spks = [spks[_] for _ in stable_pyr]

        maze_bin = np.arange(maze[0], maze[1], 0.250)
        post_bin = np.arange(post[0], post[1], 0.250)

        pre_spikecount = np.array([np.histogram(x, bins=pre_bin)[0] for x in spks])
        pre_spikecount = [
            pre_spikecount[:, i : i + windowSize]
            for i in range(0, 3 * windowSize, windowSize)
        ]
        maze_spikecount = np.array([np.histogram(x, bins=maze_bin)[0] for x in spks])
        post_spikecount = np.array([np.histogram(x, bins=post_bin)[0] for x in spks])
        post_spikecount = [
            post_spikecount[:, i : i + windowSize]
            for i in range(0, 40 * windowSize, windowSize)
        ]

        # --- pre_corr = np.corrcoef(pre_spikecount)
        pre_corr = [np.corrcoef(pre_spikecount[x]) for x in range(len(pre_spikecount))]
        maze_corr = np.corrcoef(maze_spikecount)
        post_corr = [
            np.corrcoef(post_spikecount[x]) for x in range(len(post_spikecount))
        ]

        # --- selecting only pairwise correlations from different shanks
        shnkId = np.asarray(self._obj.spikes.info.shank)
        shnkId = shnkId[stable_pyr]
        assert len(shnkId) == len(spks)
        cross_shnks = np.nonzero(np.tril(shnkId.reshape(-1, 1) - shnkId.reshape(1, -1)))
        pre_corr = [pre_corr[x][cross_shnks] for x in range(len(pre_spikecount))]
        maze_corr = maze_corr[cross_shnks]
        post_corr = [post_corr[x][cross_shnks] for x in range(len(post_spikecount))]
        print(maze_corr)

        corr_all = []
        for window in range(len(post_corr)):
            nas = np.logical_or(np.isnan(maze_corr), np.isnan(post_corr[window]))
            corr_all.append(np.corrcoef(post_corr[window][~nas], maze_corr[~nas])[0, 1])

        return np.asarray(corr_all)

    def oneD(self):
        pass

    def twoD(self):
        pass


class ExplainedVariance:

    nChans = 16
    binSize = 0.250  # in seconds
    timeWindow = 3600  # in seconds

    def __init__(self, obj):
        self._obj = obj

    def compute(self, template, matchWind, control):
        """ Calucate explained variance (EV) and reverse EV
        References:
        1) Kudrimoti 1999
        2) Tastsuno et al. 2007
        """

        pre = self._obj.epochs.pre
        maze = self._obj.epochs.maze
        post = self._obj.epochs.post

        spks = self._obj.spikes.times
        stability = self._obj.spikes.stability.info
        stable_pyr = np.where((stability.q < 4) & (stability.stable == 1))[0]
        print(f"Calculating EV for {len(stable_pyr)} stable cells")

        windowSize = self.timeWindow
        spks = [spks[_] for _ in stable_pyr]

        pre_bin = np.arange(pre[0], pre[1], self.binSize)
        maze_bin = np.arange(maze[0], maze[1], self.binSize)
        post_bin = np.arange(post[0], post[1], self.binSize)

        pre_spikecount = np.array([np.histogram(x, bins=pre_bin)[0] for x in spks])
        pre_spikecount = [
            pre_spikecount[:, i : i + windowSize]
            for i in range(0, 3 * windowSize, windowSize)
        ]
        maze_spikecount = np.array([np.histogram(x, bins=maze_bin)[0] for x in spks])
        post_spikecount = np.array([np.histogram(x, bins=post_bin)[0] for x in spks])
        post_spikecount = [
            post_spikecount[:, i : i + windowSize]
            for i in range(0, 40 * windowSize, windowSize)
        ]

        # pre_corr = np.corrcoef(pre_spikecount)
        pre_corr = [np.corrcoef(pre_spikecount[x]) for x in range(len(pre_spikecount))]
        maze_corr = np.corrcoef(maze_spikecount)
        post_corr = [
            np.corrcoef(post_spikecount[x]) for x in range(len(post_spikecount))
        ]

        # --- selecting only pairwise correlations from different shanks
        shnkId = np.asarray(self._obj.spikes.info.shank)
        shnkId = shnkId[stable_pyr]
        assert len(shnkId) == len(spks)
        cross_shnks = np.nonzero(np.tril(shnkId.reshape(-1, 1) - shnkId.reshape(1, -1)))
        # pre_corr = pre_corr[cross_shnks]
        pre_corr = [pre_corr[x][cross_shnks] for x in range(len(pre_spikecount))]
        print(len(pre_spikecount))
        maze_corr = maze_corr[cross_shnks]
        post_corr = [post_corr[x][cross_shnks] for x in range(len(post_spikecount))]

        parcorr_maze_vs_post, rev_corr = parcorr_mult([maze_corr], post_corr, pre_corr)

        ev_maze_vs_post = parcorr_maze_vs_post ** 2
        rev_corr = rev_corr ** 2

        self.ev = ev_maze_vs_post
        self.rev = rev_corr

    def plot(self, ax=None):

        ev_mean = np.mean(self.ev.squeeze(), axis=0)
        ev_std = np.std(self.ev.squeeze(), axis=0)
        rev_mean = np.mean(self.rev.squeeze(), axis=0)
        rev_std = np.std(self.rev.squeeze(), axis=0)

        if ax is None:
            plt.clf()
            fig = plt.figure(1, figsize=(10, 15))
            gs = gridspec.GridSpec(1, 1, figure=fig)
            fig.subplots_adjust(hspace=0.3)
            ax = fig.add_subplot(gs[0])

        t = (np.linspace(0, 40, 41) * 0.25)[1:] - 0.125

        ax.fill_between(
            t, ev_mean - ev_std, ev_mean + ev_std, color="#7c7979",
        )
        ax.fill_between(
            t, rev_mean - rev_std, rev_mean + rev_std, color="#87d498",
        )

        ax.plot(t,ev_meean, axis=0), "k")
        ax.plot(t, rev_mean, "#02c59b")
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Explained variance")
        ax.legend(["EV", "REV"])
        ax.text(0.2, 0.28, "POST SD", fontweight="bold")
        ax.set_xlim([0, 10])


class CellAssembly:
    def __init__(self, obj):
        pass

    def detect(self):
        pass

    def assemblyRusso(self):
        """
            This code implements Russo2017 elife paper for replay detection.
        """

        pass
