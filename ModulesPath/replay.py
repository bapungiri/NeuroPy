import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from mathutil import parcorr_mult


class Replay:

    nChans = 16
    sRate = 30000
    binSize = 0.250  # in seconds
    timeWindow = 3600  # in seconds

    def __init__(self, obj):
        self._obj = obj

    def expvar(self):
        """
            This code implements Russo2017 elife paper for replay detection.
        """

        pre = self._obj.epochs.pre  # in seconds
        maze = self._obj.epochs.maze  # in seconds
        post = self._obj.epochs.post  # in seconds
        # print(self._obj.spikes.epochs.pre)
        recording_dur = post[1]
        spks = self._obj.spikes.spks
        unstable_units = self._obj.spikes.stability.unstable
        stable_units = self._obj.spikes.stability.stable
        stable_units = list(range(len(spks)))

        spks = [spks[x] for x in stable_units]

        nUnits = len(spks)
        windowSize = self.timeWindow

        # Calculating firing rate

        # f_rate = [len(x) / recording_dur for x in spks]
        # pyr_id = [x for x in range(len(f_rate))]

        # spkAll = [spks[x] for x in pyr_id]

        pre_bin = np.arange(pre[0], pre[1], 0.250)
        maze_bin = np.arange(maze[0], maze[1], 0.250)
        post_bin = np.arange(post[0], post[1], 0.250)

        pre_spikecount = np.array([np.histogram(x, bins=pre_bin)[0] for x in spks])
        maze_spikecount = np.array([np.histogram(x, bins=maze_bin)[0] for x in spks])
        post_spikecount = np.array([np.histogram(x, bins=post_bin)[0] for x in spks])
        post_spikecount = [
            post_spikecount[:, i : i + windowSize]
            for i in range(0, 40 * windowSize, windowSize)
        ]

        pre_corr = np.corrcoef(pre_spikecount)
        maze_corr = np.corrcoef(maze_spikecount)
        post_corr = [
            np.corrcoef(post_spikecount[x]) for x in range(len(post_spikecount))
        ]

        # selecting only pairwise correlations from different shanks
        shnkId = self._obj.spikes.shankID
        shnkId = shnkId[stable_units]
        cross_shnks = np.nonzero(np.tril(shnkId.reshape(-1, 1) - shnkId.reshape(1, -1)))
        pre_corr = pre_corr[cross_shnks]
        maze_corr = maze_corr[cross_shnks]
        post_corr = [post_corr[x][cross_shnks] for x in range(len(post_spikecount))]

        # self.check2 = post_corr
        parcorr_maze_vs_post, rev_corr = parcorr_mult(
            [maze_corr], post_corr, [pre_corr]
        )

        ev_maze_vs_post = parcorr_maze_vs_post ** 2
        rev_corr = rev_corr ** 2
        return ev_maze_vs_post, rev_corr
        # self.rev = [x ** 2 for x in revcorr]
        # np.save(self.basePath + self.subname + "_EV.npy", self.ev_maze_vs_post)
        # np.save(self.basePath + self.subname + "_REV.npy", self.rev)

    def smoothEV(self):
        pass

    def bayesian1d(self):
        pass

    def bayesian2d(self):
        pass

    def assemblyICA(self):
        pass

    def assemblyRusso(self):
        """
            This code implements Russo2017 elife paper for replay detection.
        """

        pass
