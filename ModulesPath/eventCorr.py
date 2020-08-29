import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.ndimage as filtSig

import scipy.stats as stat
import matplotlib.gridspec as gridspec
from numpy.fft import fft
import matplotlib as mpl
import time
import scipy.signal as sg
from ccg import correlograms

# mpl.interactive(True)


mpl.style.use("figPublish")
from signal_process import filter_sig as filt


def psth(event_reference, event_post, trange, nbins=150):
    """
    calcualtes psth for event_post with respect to event_reference with trange around the reference event 

    event_reference = stimulus timings in seconds
    event_post = events whose occurences will be calculated w.r.t reference
    trange = [seconds before , seconds after] 
    """
    bins_refer = [
        np.linspace(x - trange[0], x + trange[1], nbins) for x in event_reference
    ]
    t_hist = np.linspace(-trange[0], trange[1], nbins)

    ripple_co = [np.histogram(event_post, bins=x)[0] for x in bins_refer]
    ripple_co = np.asarray(ripple_co)

    histall = np.sum(ripple_co)
    count_per_event = ripple_co

    return histall, count_per_event, t_hist


class event_event:
    def __init__(self, obj):

        self.hswa_ripple = hswa_ripple(obj)
        self.hswa_spindle = Hswa_spindle(obj)

        self._obj = obj

    def compute(self, ref, event, quantparam, binsize=0.01, window=1, nQuantiles=10):
        """psth of 'event' with respect to 'ref'

        Args:
            ref (array): 1-D array of timings of reference event in seconds
            event (1D array): timings of events whose psth will be calculated
            quantparam (1D array): values used to divide 'ref' into quantiles
            binsize (float, optional): [description]. Defaults to 0.01.
            window (int, optional): [description]. Defaults to 1.
            nQuantiles (int, optional): [description]. Defaults to 10.

        Returns:
            [type]: [description]
        """

        # --- parameters----------

        quantiles = pd.qcut(quantparam, nQuantiles, labels=False)

        quants, eventid = [], []
        for category in range(nQuantiles):
            indx = np.where(quantiles == category)[0]
            quants.append(ref[indx])
            eventid.append(category * np.ones(len(indx)).astype(int))

        quants.append(event)
        eventid.append(((nQuantiles + 1) * np.ones(len(event))).astype(int))

        quants = np.concatenate(quants)
        eventid = np.concatenate(eventid)

        sort_ind = np.argsort(quants)

        ccg = correlograms(
            quants[sort_ind],
            eventid[sort_ind],
            sample_rate=1250,
            bin_size=binsize,
            window_size=window,
        )

        self.psth = ccg[:-1, -1, :]

        return self.psth

    def plot(self, ax=None):

        if self.psth.ndim == 1:
            psth = self.psth[np.newaxis, :]
        else:
            psth = self.psth

        nQuantiles = self.psth.shape[0]
        cmap = mpl.cm.get_cmap("viridis")
        colmap = [cmap(x) for x in np.linspace(0, 1, nQuantiles)]

        if ax is None:
            plt.clf()
            fig = plt.figure(num=None, figsize=(10, 15))
            gs = gridspec.GridSpec(1, 1, figure=fig)
            fig.subplots_adjust(hspace=0.3)
            ax = fig.add_subplot(gs[0])

        for quant in range(nQuantiles):
            ax.plot(psth[quant, :], color=colmap[quant])
        ax.set_xlabel("Time from hswa (s)")
        ax.set_ylabel("Counts")
        ax.set_title(self._obj.sessinfo.session.sessionName)


class Hswa_spindle:
    def __init__(self, obj):
        self._obj = obj

    def compute(self, period=None, binsize=0.01, window=1, nQuantiles=10):
        """
        calculating the psth for ripple and slow wave oscillation and making n quantiles for plotting 
        """

        # --- parameters----------

        spindle = self._obj.spindle.time[:, 0]
        swa_amp = self._obj.swa.peakamp
        swa_time = self._obj.swa.time

        if period is not None:
            spindle = spindle[(spindle > period[0]) & (spindle < period[1])]
            swa_amp = swa_amp[(swa_time > period[0]) & (swa_time < period[1])]
            swa_time = swa_time[(swa_time > period[0]) & (swa_time < period[1])]

        quantiles = pd.qcut(swa_amp, nQuantiles, labels=False)

        swa_quants, eventid = [], []
        for category in range(nQuantiles):
            indx = np.where(quantiles == category)[0]
            swa_quants.append(swa_time[indx])
            eventid.append(category * np.ones(len(indx)).astype(int))

        swa_quants.append(spindle)
        eventid.append(((nQuantiles + 1) * np.ones(len(spindle))).astype(int))

        swa_quants = np.concatenate(swa_quants)
        eventid = np.concatenate(eventid)

        sort_ind = np.argsort(swa_quants)

        ccg = correlograms(
            swa_quants[sort_ind],
            eventid[sort_ind],
            sample_rate=1250,
            bin_size=binsize,
            window_size=window,
        )

        self.psth = ccg[:-1, -1, :]

        return self.psth


class hswa_ripple:
    def __init__(self, obj):
        self._obj = obj

    def compute(self, period=None, binsize=0.01, window=1, nQuantiles=10):
        """
        calculating the psth for ripple and slow wave oscillation and making n quantiles for plotting 
        """

        # --- parameters----------

        ripple = self._obj.ripple.time[:, 0]
        swa_amp = self._obj.swa.peakamp
        swa_time = self._obj.swa.time

        if period is not None:
            ripple = ripple[(ripple > period[0]) & (ripple < period[1])]
            swa_amp = swa_amp[(swa_time > period[0]) & (swa_time < period[1])]
            swa_time = swa_time[(swa_time > period[0]) & (swa_time < period[1])]

        quantiles = pd.qcut(swa_amp, nQuantiles, labels=False)

        swa_quants, eventid = [], []
        for category in range(nQuantiles):
            indx = np.where(quantiles == category)[0]
            swa_quants.append(swa_time[indx])
            eventid.append(category * np.ones(len(indx)).astype(int))

        swa_quants.append(ripple)
        eventid.append(((nQuantiles + 1) * np.ones(len(ripple))).astype(int))

        swa_quants = np.concatenate(swa_quants)
        eventid = np.concatenate(eventid)

        sort_ind = np.argsort(swa_quants)

        ccg = correlograms(
            swa_quants[sort_ind],
            eventid[sort_ind],
            sample_rate=1250,
            bin_size=binsize,
            window_size=window,
        )

        self.psth = ccg[:-1, -1, :]

        return self.psth

    def plot(self, ax=None):

        if self.psth.ndim == 1:
            psth = self.psth[np.newaxis, :]
        else:
            psth = self.psth

        nQuantiles = self.psth.shape[0]
        cmap = mpl.cm.get_cmap("viridis")
        colmap = [cmap(x) for x in np.linspace(0, 1, nQuantiles)]

        if ax is None:
            plt.clf()
            fig = plt.figure(num=None, figsize=(10, 15))
            gs = gridspec.GridSpec(1, 1, figure=fig)
            fig.subplots_adjust(hspace=0.3)
            ax = fig.add_subplot(gs[0])

        for quant in range(nQuantiles):
            ax.plot(psth[quant, :], color=colmap[quant])
        ax.set_xlabel("Time from hswa (s)")
        ax.set_ylabel("Counts")

    def plot_raster(self, ax=None):
        pass

    def plot_rippleProb(self, ax=None):
        pass
