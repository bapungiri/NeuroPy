import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


class Replay:

    nChans = 16
    sRate = 30000
    binSize = 0.250  # in seconds
    timeWindow = 3600  # in seconds

    def __init__(self, obj):
        self._obj = obj

    def EV(self):

        epoch_time = np.load(self.filePrefix + "_epochs.npy", allow_pickle=True)
        recording_dur = epoch_time.item().get("POST")[1]  # in seconds
        pre = epoch_time.item().get("PRE")  # in seconds
        maze = epoch_time.item().get("MAZE")  # in seconds
        post = epoch_time.item().get("POST")  # in seconds

        self.nUnits = len(self.spkAll)
        windowSize = self.timeWindow

        # Calculating firing rate
        f_rate = [len(x) / recording_dur for x in self.spkAll]
        pyr_id = [x for x in range(len(f_rate)) if f_rate[x] < 10]
        spkAll = [self.spkAll[x] for x in pyr_id]

        pre_bin = np.arange(pre[0], pre[1], 0.250)
        maze_bin = np.arange(maze[0], maze[1], 0.250)
        post_bin = np.arange(post[0], post[1], 0.250)

        pre_spikecount = np.array([np.histogram(x, bins=pre_bin)[0] for x in spkAll])
        maze_spikecount = np.array([np.histogram(x, bins=maze_bin)[0] for x in spkAll])
        post_spikecount = np.array([np.histogram(x, bins=post_bin)[0] for x in spkAll])
        post_spikecount = [
            post_spikecount[:, i : i + windowSize]
            for i in range(0, 10 * windowSize, windowSize)
        ]

        pre_corr = np.corrcoef(pre_spikecount)
        maze_corr = np.corrcoef(maze_spikecount)
        post_corr = [
            np.corrcoef(post_spikecount[x]) for x in range(len(post_spikecount))
        ]

        self.check1 = post_corr
        # selecting only lower diagonal of matrix
        pre_corr = pre_corr[np.tril_indices(pre_corr.shape[0], k=-1)]
        maze_corr = maze_corr[np.tril_indices(maze_corr.shape[0], k=-1)]
        post_corr = [
            post_corr[x][np.tril_indices(post_corr[x].shape[0], k=-1)]
            for x in range(len(post_spikecount))
        ]

        self.check2 = post_corr
        parcorr_maze_vs_post, revcorr = self.partialCorrelation(
            maze_corr, post_corr, pre_corr
        )

        self.ev_maze_vs_post = [x ** 2 for x in parcorr_maze_vs_post]
        self.rev = [x ** 2 for x in revcorr]
        np.save(self.basePath + self.subname + "_EV.npy", self.ev_maze_vs_post)
        np.save(self.basePath + self.subname + "_REV.npy", self.rev)

    def smoothEV(self):

        epoch_time = np.load(self.filePrefix + "_epochs.npy", allow_pickle=True)
        recording_dur = epoch_time.item().get("POST")[1]  # in seconds
        pre = epoch_time.item().get("PRE")  # in seconds
        maze = epoch_time.item().get("MAZE")  # in seconds
        post = epoch_time.item().get("POST")  # in seconds

        self.nUnits = len(self.spkAll)
        windowSize = self.timeWindow

        # Calculating firing rate
        f_rate = [len(x) / recording_dur for x in self.spkAll]
        pyr_id = [x for x in range(len(f_rate)) if f_rate[x] < 10]
        spkAll = [self.spkAll[x] for x in pyr_id]

        pre_bin = np.arange(pre[0], pre[1], 0.250)
        maze_bin = np.arange(maze[0], maze[1], 0.250)
        post_bin = np.arange(post[0], post[1], 0.250)

        pre_spikecount = np.array([np.histogram(x, bins=pre_bin)[0] for x in spkAll])
        maze_spikecount = np.array([np.histogram(x, bins=maze_bin)[0] for x in spkAll])
        post_spikecount = np.array([np.histogram(x, bins=post_bin)[0] for x in spkAll])

        post_spikecount = [
            post_spikecount[:, i : i + windowSize]
            for i in range(0, 10 * windowSize, int(windowSize / 3))
        ]

        pre_corr = np.corrcoef(pre_spikecount)
        maze_corr = np.corrcoef(maze_spikecount)
        post_corr = [
            np.corrcoef(post_spikecount[x]) for x in range(len(post_spikecount))
        ]

        self.check1 = post_corr
        # selecting only lower diagonal of matrix
        pre_corr = pre_corr[np.tril_indices(pre_corr.shape[0], k=-1)]
        maze_corr = maze_corr[np.tril_indices(maze_corr.shape[0], k=-1)]
        post_corr = [
            post_corr[x][np.tril_indices(post_corr[x].shape[0], k=-1)]
            for x in range(len(post_spikecount))
        ]

        self.check2 = post_corr
        parcorr_maze_vs_post, revcorr = self.partialCorrelation(
            maze_corr, post_corr, pre_corr
        )

        self.ev_maze_vs_post = [x ** 2 for x in parcorr_maze_vs_post]
        self.rev = [x ** 2 for x in revcorr]

        np.save(self.basePath + self.subname + "_EV.npy", self.ev_maze_vs_post)
        np.save(self.basePath + self.subname + "_REV.npy", self.rev)
