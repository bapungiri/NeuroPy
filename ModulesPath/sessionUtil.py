import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import scipy.stats as stats
import scipy.signal as sg
from pathlib import Path
import matplotlib.gridspec as gridspec
import signal_process
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from sklearn.preprocessing import normalize
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import matplotlib as mpl
from mathutil import threshPeriods


class SessionUtil:

    binSize = 0.001  # in seconds
    gauss_std = 0.025  # in seconds

    def __init__(self, obj):
        self._obj = obj

    def geteeg(self, chans, timeRange=None, frames=None):
        """Returns eeg signal for given channels and timeperiod or selected frames

        Args:
            chans (list): list of channels required index should in order of binary file
            timeRange (list, optional): In seconds and must have length 2.
            frames (list, optional): Required frames from the eeg data.

        Returns:
            eeg: [array of channels x timepoints]
        """
        eegfile = self._obj.sessinfo.recfiles.eegfile
        eegSrate = self._obj.recinfo.lfpSrate
        nChans = self._obj.recinfo.nChans

        eeg = np.memmap(eegfile, dtype="int16", mode="r")
        eeg = np.memmap.reshape(eeg, (nChans, len(eeg) // nChans), order="F")

        if timeRange is not None:
            assert len(timeRange) == 2
            frameStart = int(timeRange[0] * eegSrate)
            frameEnd = int(timeRange[1] * eegSrate)
            eeg = eeg[:, frameStart:frameEnd]
        elif frames is not None:
            eeg = eeg[:, frames]

        eeg = eeg[chans, :]
        return eeg

    def plotChanPos(self, chans=None, ax=None, colors=None):

        nShanks = self._obj.recinfo.nShanks
        channelgrp = self._obj.recinfo.channelgroups[:nShanks]
        lfpchans = np.asarray([chan for shank in channelgrp for chan in shank])

        chans2plot = chans
        chan_rank = np.where(np.isin(lfpchans, chans2plot))[0]
        xpos, ypos = self._obj.recinfo.probemap()
        xpos = np.asarray(xpos)
        ypos = np.asarray(ypos)

        if ax is None:
            fig = plt.figure(1, figsize=(10, 15))
            gs = gridspec.GridSpec(1, 1, figure=fig)
            fig.subplots_adjust(hspace=0.3)
            ax = fig.add_subplot(gs[0])

        ax.scatter(xpos, ypos, s=4, color="gray", zorder=1)
        if colors is None:
            ax.scatter(xpos[chan_rank], ypos[chan_rank], c="red", s=20, zorder=2)
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

    def strong_theta_lfp(self, chans, period):

        assert len(period) == 2
        tstart, tend = period

        eegSrate = self._obj.recinfo.lfpSrate
        lfp, _, _ = self._obj.spindle.best_chan_lfp()
        t = np.linspace(0, len(lfp) / eegSrate, len(lfp))

        lfpperiod = lfp[(t > tstart) & (t < tend)]
        tperiod = np.linspace(tstart, tend, len(lfpperiod))

        frtheta = np.arange(5, 12, 0.5)
        wavdec = signal_process.wavelet_decomp(lfpperiod, freqs=frtheta)
        wav = wavdec.cohen(ncycles=7)

        sum_theta = gaussian_filter1d(np.sum(wav, axis=0), sigma=10)
        zsc_theta = stats.zscore(sum_theta)
        thetaevents = threshPeriods(
            zsc_theta,
            lowthresh=0,
            highthresh=1.5,
            minDistance=625,
            minDuration=2 * 1250,
        )
        thetaevents = thetaevents / eegSrate + tstart

        lfp_ind = np.concatenate(
            [
                np.arange(int(beg * eegSrate), int((end) * eegSrate))
                for (beg, end) in thetaevents
            ]
        )

        chanlfp = self.geteeg(chans=chans)
        lfp_theta = chanlfp[:, lfp_ind]

        return lfp_theta

    def getinterval(self, period, nbins):

        interval = np.linspace(period[0], period[1], nbins + 1)
        interval = [[interval[i], interval[i + 1]] for i in range(nbins)]
        return interval

    def spectrogram(self, period=None, freqRange=None):

        eegSrate = self._obj.recinfo.lfpSrate
        if freqRange is None:
            freqRange = [0, eegSrate / 2]
        if period is None:
            lfp = self.geteeg(chans=self._obj.theta.bestchan)

        specgram = signal_process.spectrogramBands(
            lfp, sampfreq=eegSrate, window=4, overlap=2
        )

        return specgram

