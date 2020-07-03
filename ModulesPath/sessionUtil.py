import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import scipy.stats as stats
import scipy.signal as sg
from pathlib import Path
import matplotlib.gridspec as gridspec
from signal_process import spectrogramBands
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import normalize
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import matplotlib as mpl


class SessionUtil:

    binSize = 0.001  # in seconds
    gauss_std = 0.025  # in seconds

    def __init__(self, obj):
        self._obj = obj

    def geteeg(self, channels, timeRange=None):
        """Returns eeg signal for given channels and timeperiod

        Args:
            channels (list): list of channels required index should in order of binary file
            timeRange (list, optional): In seconds and must have length 2.

        Returns:
            eeg: [array of channels x timepoints]
        """
        eegfile = self._obj.sessinfo.recfiles.eegfile
        eegSrate = self._obj.recinfo.lfpSrate
        nChans = self._obj.recinfo.nChans

        if timeRange is None:
            eeg = np.memmap(eegfile, dtype="int16", mode="r")
            eeg = np.memmap.reshape(eeg, (int(len(eeg) / nChans), nChans))
            eeg = eeg[:, channels].T

        else:

            assert len(timeRange) == 2

            frameStart = int(timeRange[0] * eegSrate)
            frameEnd = int(timeRange[1] * eegSrate)

            eeg = np.memmap(
                eegfile,
                dtype="int16",
                offset=2 * frameStart * nChans,
                shape=((frameEnd - frameStart), nChans),
                mode="r",
            )

            eeg = np.asarray(eeg[:, channels].T)

        return eeg

    def plotChanPos(self, chans=None, ax=None, colors=None):

        nShanks = self._obj.recinfo.nShanks
        channelgrp = self._obj.recinfo.channelgroups[:nShanks]
        lfpchans = [chan for shank in channelgrp for chan in shank]

        chans2plot = chans
        chan_rank = [
            order for order in range(len(lfpchans)) if lfpchans[order] in chans2plot
        ]
        xpos, ypos = self._obj.recinfo.probemap()
        xpos = np.asarray(xpos)
        ypos = np.asarray(ypos)

        if ax is None:
            fig = plt.figure(1, figsize=(10, 15))
            gs = gridspec.GridSpec(1, 1, figure=fig)
            fig.subplots_adjust(hspace=0.3)
            ax = fig.add_subplot(gs[0])

        ax.plot(xpos, ypos, ".", color="gray", zorder=1)
        if colors is None:
            ax.plot(xpos[chan_rank], ypos[chan_rank], "r.")
        else:
            ax.scatter(xpos[chan_rank], ypos[chan_rank], c=colors, s=40, zorder=2)

    def export2Neuroscope(self, times, suffix="evt"):
        times = times * 1000  # convert to ms
        file_neuroscope = self._obj.sessinfo.files.filePrefix.with_suffix(
            f".evt.{suffix}"
        )
        with file_neuroscope.open("w") as a:
            for beg, stop in times:
                a.write(f"{beg} start\n{stop} end\n")
