import os
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d
from sklearn.naive_bayes import GaussianNB
import math


class DecodeBehav:
    nShanks = 8

    def __init__(self, obj):

        # self._obj = obj
        self.bayes1d = bayes1d(obj)
        self.bayes2d = bayes2d(obj)


class bayes1d:
    def __init__(self, obj):

        self._obj = obj

    def fit(self):
        pass

    def predict(self):
        pass


class bayes2d:
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
        ind_maze = np.where((t > maze[0]) & (t < maze[1]))
        x = x[ind_maze]
        y = y[ind_maze]
        t = t[ind_maze]

        x = x + abs(min(x))
        y = y + abs(min(y))

        x_grid = np.arange(min(x), max(x), 10)
        y_grid = np.arange(min(y), max(y), 10)
        x_, y_ = np.meshgrid(x_grid, y_grid)

        diff_posx = np.diff(x)
        diff_posy = np.diff(y)

        speed = np.sqrt(diff_posx ** 2 + diff_posy ** 2)
        dt = t[1] - t[0]
        speed_thresh = np.where(speed / dt > 0)[0]

        x_thresh = x[speed_thresh]
        y_thresh = y[speed_thresh]
        t_thresh = t[speed_thresh]

        occupancy = np.histogram2d(x, y, bins=(x_grid, y_grid))[0]
        shape_occ = occupancy.shape
        occupancy = occupancy + np.spacing(1)
        occupancy = occupancy / 120  # converting to seconds
        # plt.subplot(1, 2, 1)
        # plt.imshow(occupancy.T, origin="lower")
        # plt.subplot(1, 2, 2)
        # plt.plot(x, y)

        linear_pos = occupancy.flatten()
        pos_ind = np.arange(len(linear_pos))

        bin_t = np.arange(t[0], t[-1], 0.1)
        x_bin = np.interp(bin_t, t, x)
        y_bin = np.interp(bin_t, t, y)

        _, _, _, bin_number_t = binned_statistic_2d(
            x_bin, y_bin, x_bin, "count", bins=[x_grid, y_grid], expand_binnumbers=True,
        )

        print(np.min(bin_number_t - 1, axis=1))
        print(shape_occ)

        bin_number_t = np.ravel_multi_index(bin_number_t - 1, shape_occ)

        spkcount = np.asarray([np.histogram(x, bins=bin_t)[0] for x in spkAll])

        # ======== using sklearn bayes classifier didn't work ================
        # gnb = GaussianNB()
        # y_pred = gnb.fit(spkcount, bin_number_t[:-1]).predict(spkcount)

        # plt.plot(bin_number_t, "k")
        # plt.plot(y_pred, "r")
        # for x_, y_ in zip(x_bin, y_bin):
        #     occ_bin = np.histogram2d(x, y, bins=(x_grid, y_grid))[0]

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
        print(ratemap.shape)

        ntbin = len(bin_t)
        nposbin = len(linear_pos)
        prob = (
            lambda nspike, rate: (1 / math.factorial(nspike))
            * ((0.1 * rate) ** nspike)
            * (np.exp(-0.1 * rate))
        )

        pos_decode = []
        for timebin in range(100):
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

    def predict(self):
        pass
