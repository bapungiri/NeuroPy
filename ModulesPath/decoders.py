import os
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d
from sklearn.naive_bayes import GaussianNB


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

        # we require only maze portion
        ind_maze = np.where((t > maze[0] + 60) & (t < maze[1] - 90))
        x = x[ind_maze]
        y = y[ind_maze]
        t = t[ind_maze]

        x = x + abs(min(x))
        y = y + abs(min(y))

        x_grid = np.arange(min(x), max(x), 30)
        y_grid = np.arange(min(y), max(y), 30)
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
            x_bin, y_bin, x_bin, "count", bins=(x_grid, y_grid)
        )

        spkcount = np.asarray([np.histogram(x, bins=bin_t)[0] for x in spkAll]).T
        # gnb = GaussianNB()
        # y_pred = gnb.fit(spkcount, bin_number_t[:-1]).predict(spkcount)

        # plt.plot(bin_number_t, "k")
        # plt.plot(y_pred, "r")
        # for x_, y_ in zip(x_bin, y_bin):
        #     occ_bin = np.histogram2d(x, y, bins=(x_grid, y_grid))[0]

        # pf, spk_pos = [], []
        # for cell in spkAll:

        #     spk_maze = cell[np.where((cell > maze[0]) & (cell < maze[1]))]
        #     spk_speed = np.interp(spk_maze, t[1:], speed)
        #     spk_y = np.interp(spk_maze, t, y)
        #     spk_x = np.interp(spk_maze, t, x)

        #     # speed threshold
        #     # spd_ind = np.where(spk_speed > 2)
        #     # spk_spd = spk_speed[spd_ind]
        #     # spk_x = spk_x[spd_ind]
        #     # spk_y = spk_y[spd_ind]
        #     # spk_t = spk_maze[spd_ind]

        #     spk_map = np.histogram2d(spk_x, spk_y, bins=(x_grid, y_grid))[0]
        #     pf.append(spk_map / occupancy)
        #     spk_pos.append([spk_x, spk_y])

        # for pos_bin in pos_ind:
        # posterior =

    def predict(self):
        pass
