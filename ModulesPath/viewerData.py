import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import scipy.stats as stats
import scipy.signal as sg
from pathlib import Path
from matplotlib.gridspec import GridSpec
from signal_process import spectrogramBands
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import normalize
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from visbrain.gui import Sleep


def make_boxes(
    ax, xdata, ydata, xerror, yerror, facecolor="r", edgecolor="None", alpha=0.5
):

    # Loop over data points; create box from errors at each point
    errorboxes = [
        Rectangle((x, y), xe, ye) for x, y, xe, ye in zip(xdata, ydata, xerror, yerror)
    ]

    # Create patch collection with specified colour/alpha
    pc = PatchCollection(
        errorboxes, facecolor=facecolor, alpha=alpha, edgecolor=edgecolor
    )

    # Add collection to axes
    ax.add_collection(pc)

    return 1


class SessView:

    binSize = 0.001  # in seconds
    gauss_std = 0.025  # in seconds

    def __init__(self, obj):
        self._obj = obj

    def specgram(self, ax=None):
        lfp, _, _ = self._obj.spindle.best_chan_lfp()
        lfpSrate = self._obj.recinfo.lfpSrate
        spec = spectrogramBands(lfp, window=5 * lfpSrate)
        sxx = spec.sxx / np.max(spec.sxx)
        sxx = gaussian_filter(sxx, sigma=1)
        print(np.max(sxx), np.min(sxx))
        vmax = np.max(sxx) / 1000

        if ax is None:
            _, ax = plt.subplots(1, 1)

        ax.pcolorfast(spec.time, spec.freq, sxx, cmap="YlGn", vmax=vmax)
        ax.set_ylim([0, 60])
        ax.set_xlim([np.min(spec.time), np.max(spec.time)])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (s)")

    def epoch(self, ax=None):
        epochs = self._obj.epochs.times

        if ax is None:
            ax = plt.subplots(1, 1)

        ind = -1
        for col in epochs.columns:
            period = epochs[col].values
            ax.fill_between(
                period,
                (ind + 1) * np.ones(len(period)),
                ind * np.ones(len(period)),
                color="#1DE9B6",
            )
            ind = ind - 1
        ax.set_yticks([-0.5, -1.5, -2.5])
        ax.set_yticklabels(["pre", "maze", "post"])
        ax.spines["left"].set_visible(False)
        # ax.set_xticklabels([""])

    def position(self):
        pass

    def raster(self, ax=None):
        spikes = self._obj.spikes.times

        if ax is None:
            ax = plt.subplots(1, 1)

        for cell, spk in enumerate(spikes):
            plt.plot(spk, cell * np.ones(len(spk)), "|", markersize=1)

    def brainstates(self, ax1=None):
        states = self._obj.brainstates.states

        if ax1 is None:
            fig = plt.figure(1, figsize=(6, 10))
            gs = GridSpec(9, 1, figure=fig)
            fig.subplots_adjust(hspace=0.4)
            ax1 = fig.add_subplot(gs[0, 0])

        x = np.asarray(states.start)
        y = np.zeros(len(x)) + np.asarray(states.state)
        width = np.asarray(states.duration)
        height = np.ones(len(x))
        qual = states.state

        colors = ["#6b90d1", "#eb9494", "#b6afaf", "#474343"]
        col = [colors[int(state) - 1] for state in states.state]

        make_boxes(ax1, x, y, width, height, facecolor=col)
        ax1.set_ylim(1, 5)

    def lfpevents(self, ax=None):
        ripples = self._obj.ripple.time
        spindles = self._obj.spindle.time

        if ax is None:
            fig = plt.figure(1, figsize=(6, 10))
            gs = GridSpec(9, 1, figure=fig)
            fig.subplots_adjust(hspace=0.4)
            ax = fig.add_subplot(gs[0, 0])

        # colors = ["#6b90d1", "#eb9494", "#b6afaf", "#474343"]
        # col = [colors[int(state) - 1] for state in states.state]
        width = np.diff(ripples, axis=1).squeeze()
        height = 0.2 * np.ones(len(ripples))
        # ax.plot(ripples[:, 0], np.ones(len(ripples)), ".", markersize=0.5)

        make_boxes(
            ax,
            ripples[:, 0],
            np.ones(len(ripples)),
            width,
            height,
            facecolor="#eb9494",
        )
        ax.set_ylim(1, 1.2)

    def summary(self):

        fig = plt.figure(num=None, figsize=(20, 7))
        gs = GridSpec(10, 5, figure=fig)
        fig.subplots_adjust(hspace=0.5)

        ax = fig.add_subplot(gs[1:3, :])
        self.specgram(ax=ax)

        ax = fig.add_subplot(gs[0, :], sharex=ax)
        self.brainstates(ax1=ax)

        ax = fig.add_subplot(gs[3, :], sharex=ax)
        self.epoch(ax=ax)

        ax = fig.add_subplot(gs[4:6, :], sharex=ax)
        self.raster(ax=ax)

        ax = fig.add_subplot(gs[6, :], sharex=ax)
        self.lfpevents(ax=ax)

    def testsleep(self):
        lfp = self._obj.spindle.best_chan_lfp()[0]
        lfp = np.c_[lfp, lfp].T
        lfpt = np.linspace(0, len(lfp) / 1250, len(lfp))
        channels = [1, 2]

        hypno = None
        Sleep(data=lfp, hypno=hypno, channels=channels, sf=1250).show()
