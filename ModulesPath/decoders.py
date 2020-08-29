import os
from matplotlib.pyplot import axis
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d
from sklearn.naive_bayes import GaussianNB
import math
from scipy.ndimage import gaussian_filter1d
from scipy.special import factorial
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd


class DecodeBehav:
    def __init__(self, obj):

        # self._obj = obj
        self.bayes1d = bayes1d(obj)
        self.bayes2d = bayes2d(obj)


class bayes1d:
    def __init__(self, obj):
        self._obj = obj

    def fit(self):
        spkAll = self._obj.spikes.times
        x = self._obj.position.x
        y = self._obj.position.y
        t = self._obj.position.t
        maze = self._obj.epochs.maze  # in seconds
        maze[0] = maze[0] + 60
        maze[1] = maze[1] - 90

        # we require only maze portion
        ind_maze = np.where((t > maze[0]) & (t < maze[1]))[0]
        x = y[ind_maze]
        y = y[ind_maze]
        t = t[ind_maze]

        x = x + abs(min(x))
        x_grid = np.arange(min(x), max(x), 10)

        diff_posx = np.diff(x)
        diff_posy = np.diff(y)

        speed = np.sqrt(diff_posx ** 2 + diff_posy ** 2)
        dt = t[1] - t[0]
        speed_thresh = np.where(speed / dt > 0)[0]

        x_thresh = x[speed_thresh]
        y_thresh = y[speed_thresh]
        t_thresh = t[speed_thresh]

        occupancy = np.histogram(x, bins=x_grid)[0]
        shape_occ = occupancy.shape
        occupancy = occupancy + np.spacing(1)
        occupancy = occupancy / 120  # converting to seconds

        bin_t = np.arange(t[0], t[-1], 0.1)
        x_bin = np.interp(bin_t, t, x)
        y_bin = np.interp(bin_t, t, y)

        bin_number_t = np.digitize(x_bin, bins=x_grid)

        spkcount = np.asarray([np.histogram(x, bins=bin_t)[0] for x in spkAll])
        ratemap, spk_pos = [], []
        for cell in spkAll:

            spk_maze = cell[np.where((cell > maze[0]) & (cell < maze[1]))]
            spk_speed = np.interp(spk_maze, t[1:], speed)
            spk_y = np.interp(spk_maze, t, y)
            spk_x = np.interp(spk_maze, t, x)

            spk_map = np.histogram(spk_y, bins=x_grid)[0]
            spk_map = spk_map / occupancy
            ratemap.append(spk_map)
            spk_pos.append([spk_x, spk_y])

        ratemap = np.asarray(ratemap)
        print(ratemap.shape)

        ntbin = len(bin_t)
        nposbin = len(x_grid) - 1
        prob = (
            lambda nspike, rate: (1 / math.factorial(nspike))
            * ((0.1 * rate) ** nspike)
            * (np.exp(-0.1 * rate))
        )

        pos_decode = []
        for timebin in range(len(bin_t) - 1):
            spk_bin = spkcount[:, timebin]

            prob_allbin = []
            for posbin in range(nposbin):
                rate_bin = ratemap[:, posbin]
                spk_prob_bin = [prob(spk, rate) for spk, rate in zip(spk_bin, rate_bin)]
                prob_thisbin = np.prod(spk_prob_bin)
                prob_allbin.append(prob_thisbin)

            prob_allbin = np.asarray(prob_allbin)

            posterior = prob_allbin / np.sum(prob_allbin)
            predict_bin = np.argmax(posterior)

            pos_decode.append(predict_bin)

        plt.plot(bin_number_t, "k")
        plt.plot(pos_decode, "r")


class bayes2d:
    def __init__(self, obj):

        self._obj = obj

    def fit(self):
        trackingSrate = self._obj.position.tracking_sRate
        spkAll = self._obj.spikes.pyr
        x = self._obj.position.x
        y = self._obj.position.y
        t = self._obj.position.t
        maze = self._obj.epochs.maze  # in seconds

        # --- we require only maze portion -----
        ind_maze = np.where((t > maze[0]) & (t < maze[1]))
        x = x[ind_maze]
        y = y[ind_maze]
        t = t[ind_maze]

        x_grid = np.linspace(min(x), max(x), 15)
        y_grid = np.linspace(min(y), max(y), 15)

        diff_posx = np.diff(x)
        diff_posy = np.diff(y)

        speed = np.sqrt(diff_posx ** 2 + diff_posy ** 2)
        dt = t[1] - t[0]
        speed_thresh = np.where(speed / dt > 0)[0]

        occupancy = np.histogram2d(x, y, bins=(x_grid, y_grid))[0]
        occupancy = (occupancy + np.spacing(1)) / trackingSrate

        linear_pos = occupancy.flatten()

        bin_t = np.arange(t[0], t[-1], 0.2)
        x_bin = np.interp(bin_t, t, x)
        y_bin = np.interp(bin_t, t, y)
        speed_bin = np.interp(bin_t, t[1:], speed)
        print(len(speed_bin), len(x_bin))

        bin_number_t = binned_statistic_2d(
            x_bin, y_bin, x_bin, "count", bins=[x_grid, y_grid], expand_binnumbers=True,
        )[3]

        bin_number_t = np.ravel_multi_index(bin_number_t - 1, occupancy.shape)
        spkcount = np.asarray([np.histogram(x, bins=bin_t)[0] for x in spkAll])

        ratemap, spk_pos = [], []
        for cell in spkAll:

            spk_maze = cell[np.where((cell > maze[0]) & (cell < maze[1]))]
            spk_speed = np.interp(spk_maze, t[1:], speed)
            spk_y = np.interp(spk_maze, t, y)
            spk_x = np.interp(spk_maze, t, x)

            # speed threshold
            # spd_ind = np.where(spk_speed > 2)
            # spk_spd = spk_speed[spd_ind]
            # spk_x = spk_x[spd_ind]
            # spk_y = spk_y[spd_ind]
            # spk_t = spk_maze[spd_ind]

            spk_map = np.histogram2d(spk_x, spk_y, bins=(x_grid, y_grid))[0]
            spk_map = (spk_map / occupancy).flatten()
            ratemap.append(spk_map)
            spk_pos.append([spk_x, spk_y])

        ratemap = np.asarray(ratemap)

        """ 
        ===========================
        Probability is calculated using this formula
        prob = (1 / nspike!)* ((0.1 * frate)^nspike) * exp(-0.1 * frate)
        =========================== 
        """

        Ncells = len(spkAll)
        cell_prob = np.zeros((ratemap.shape[1], spkcount.shape[1], Ncells))
        for cell in range(Ncells):
            cell_spkcnt = spkcount[cell, :][np.newaxis, :]
            cell_ratemap = ratemap[cell, :][:, np.newaxis]

            coeff = 1 / (factorial(cell_spkcnt))
            # broadcasting
            cell_prob[:, :, cell] = (((0.1 * cell_ratemap) ** cell_spkcnt) * coeff) * (
                np.exp(-0.1 * cell_ratemap)
            )

        posterior = np.prod(cell_prob, axis=2)
        posterior /= np.sum(posterior, axis=0)
        self.posterior = posterior
        self.bin_number = bin_number_t
        self.decodedPos = np.argmax(self.posterior, axis=0)
        self.xgrid = x_grid
        self.ygrid = y_grid
        self.tgrid = bin_t
        self.xpos = x
        self.ypos = y
        self.t = t
        self.ratemap = ratemap
        self.speed_bin = speed_bin

    def decode(self, epochs, binsize=0.02):

        assert isinstance(epochs, pd.DataFrame)
        # TODO plot only running epochs

        running_bins = self.speed_bin > 5
        bin_number = self.bin_number.astype(float)
        bin_number[running_bins] = np.nan

        spkAll = self._obj.spikes.pyr
        self.fit()
        ratemap = self.ratemap

        nbins = np.zeros(len(epochs))
        spkcount = []
        for i, epoch in enumerate(epochs.itertuples()):
            bins = np.arange(epoch.start, epoch.end, binsize)
            nbins[i] = len(bins) - 1
            spkcount.append(np.asarray([np.histogram(_, bins=bins)[0] for _ in spkAll]))

        spkcount = np.hstack(spkcount)

        """ 
        ===========================
        Probability is calculated using this formula
        prob = (1 / nspike!)* ((0.1 * frate)^nspike) * exp(-0.1 * frate)
        =========================== 
        """

        Ncells = len(spkAll)
        cell_prob = np.zeros((ratemap.shape[1], spkcount.shape[1], Ncells))
        for cell in range(Ncells):
            cell_spkcnt = spkcount[cell, :][np.newaxis, :]
            cell_ratemap = ratemap[cell, :][:, np.newaxis]

            coeff = 1 / (factorial(cell_spkcnt))
            # broadcasting
            cell_prob[:, :, cell] = (((0.1 * cell_ratemap) ** cell_spkcnt) * coeff) * (
                np.exp(-0.1 * cell_ratemap)
            )
        posterior = np.prod(cell_prob, axis=2)
        posterior /= np.sum(posterior, axis=0)

        decodedPos = np.argmax(posterior, axis=0)
        cum_nbins = np.append(0, np.cumsum(nbins)).astype(int)

        decodedPos = [
            decodedPos[cum_nbins[i] : cum_nbins[i + 1]]
            for i in range(len(cum_nbins) - 1)
        ]

        return decodedPos

    def plot(self):

        pos_decode = gaussian_filter1d(self.decodedPos, sigma=1)
        posterior = self.posterior
        npos = posterior.shape[0]
        time = posterior.shape[1]
        pos_mat = np.arange(npos).reshape(14, 14)

        plt.clf()
        fig = plt.figure(1, figsize=(10, 15))
        gs = gridspec.GridSpec(3, 6, figure=fig)
        fig.subplots_adjust(hspace=0.3)

        ax = fig.add_subplot(gs[0, :])
        ax.pcolormesh(self.tgrid[1:], np.arange(npos), posterior, cmap="binary")
        ax.plot(self.tgrid[1:], self.bin_number[1:], "#4FC3F7")
        ax.plot(self.tgrid[1:], pos_decode, "#F48FB1")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Position (bin)")
        ax.set_title("Bayesian position estimation (only pyr cells)")

        ax = fig.add_subplot(gs[1, :], sharex=ax)
        ax.plot(self.tgrid, self.speed_bin, "k")
        ax.set_ylabel("Speed")
        ax.set_xlabel("Time (s)")

        ax = fig.add_subplot(gs[2, 0])
        rand_time = np.random.randint(self.tgrid[0], self.tgrid[-1] - 40)
        rand_time = 8797
        indx = np.where((self.tgrid > rand_time) & (self.tgrid < rand_time + 10))[0]

        a = self.bin_number[indx]
        b = pos_decode[indx]

        ax.pcolormesh(
            self.tgrid[1:][indx], np.arange(npos), posterior[:, indx], cmap="binary"
        )
        ax.plot(self.tgrid[indx], a, "#4FC3F7")
        ax.plot(self.tgrid[indx], b, "#F48FB1")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Position")

        # sns.heatmap(pos_mat, linewidths=0.5, cbar=False, ax=ax, cmap="k")
        a = self.decodedPos[indx]
        pos_ = np.unravel_index(a, (14, 14))

        ax = fig.add_subplot(gs[2, 1])
        x_ = self.xpos
        y_ = self.ypos
        indx = np.where((self.t > rand_time) & (self.t < rand_time + 10))[0]

        ax.plot(x_, y_, "k", alpha=0.8)
        ax.plot(x_[indx], y_[indx], "#4FC3F7")
        ax.plot(self.xgrid[pos_[0]] + 7, self.ygrid[pos_[1]] + 7, "#F48FB1")
        ax.set_xticks(self.xgrid)
        ax.set_xticklabels(np.arange(len(self.xgrid)))
        ax.set_yticks(self.xgrid)
        ax.set_yticklabels(np.arange(len(self.xgrid)))
        ax.set_xlabel("xbin")
        ax.set_ylabel("ybin")

        ax.grid(True)

        ax = fig.add_subplot(gs[2, 2])
        rand_time = np.random.randint(self.tgrid[0], self.tgrid[-1] - 40)
        rand_time = 9790
        indx = np.where((self.tgrid > rand_time) & (self.tgrid < rand_time + 10))[0]

        a = self.bin_number[indx]
        b = pos_decode[indx]

        ax.pcolormesh(
            self.tgrid[1:][indx], np.arange(npos), posterior[:, indx], cmap="binary"
        )
        ax.plot(self.tgrid[indx], a, "#4FC3F7")
        ax.plot(self.tgrid[indx], b, "#F48FB1")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Position")

        # sns.heatmap(pos_mat, linewidths=0.5, cbar=False, ax=ax, cmap="k")
        a = self.decodedPos[indx]
        pos_ = np.unravel_index(a, (14, 14))

        ax = fig.add_subplot(gs[2, 3])
        x_ = self.xpos
        y_ = self.ypos
        indx = np.where((self.t > rand_time) & (self.t < rand_time + 10))[0]

        ax.plot(x_, y_, "k", alpha=0.8)
        ax.plot(x_[indx], y_[indx], "#4FC3F7")
        ax.plot(self.xgrid[pos_[0]] + 7, self.ygrid[pos_[1]] + 7, "#F48FB1")
        ax.set_xticks(self.xgrid)
        ax.set_xticklabels(np.arange(len(self.xgrid)))
        ax.set_yticks(self.xgrid)
        ax.set_yticklabels(np.arange(len(self.xgrid)))
        ax.set_xlabel("xbin")
        ax.set_ylabel("ybin")

        ax.grid(True)

