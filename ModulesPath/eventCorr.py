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


class hswa_ripple:

    nQuantiles = 5

    def __init__(self, obj):
        self._obj = obj

    def compute(self, period=None, binsize=0.01, window=1):
        """
        calculating the psth for ripple and slow wave oscillation and making n quantiles for plotting 
        """

        # --- parameters----------

        ripple = self._obj.ripple.time[:, 0]
        swa = self._obj.swa.time

        if period is not None:
            ripple = ripple[(ripple > period[0]) & (ripple < period[1])]
            swa = swa[(swa > period[0]) & (swa < period[1])]

        quantiles = pd.qcut(swa, self.nQuantiles, labels=False)

        swa_quants, eventid = [], []
        for category in range(self.nQuantiles):
            indx = np.where(quantiles == category)[0]
            swa_quants.append(swa[indx])
            eventid.append(category * np.ones(len(indx)).astype(int))
            print(category)

        swa_quants.append(ripple)
        eventid.append(((category + 1) * np.ones(len(ripple))).astype(int))

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

    # plotting methods

    def plot(self, ax=None):
        cmap = mpl.cm.get_cmap("viridis")
        colmap = [cmap(x) for x in np.linspace(0, 1, self.nQuantiles)]

        if ax is None:
            plt.clf()
            fig = plt.figure(num=None, figsize=(10, 15))
            gs = gridspec.GridSpec(1, 1, figure=fig)
            fig.subplots_adjust(hspace=0.3)
            ax = fig.add_subplot(gs[0])

        for quant in range(self.nQuantiles):
            ax.plot(self.psth[quant, :], color=colmap[quant])
        ax.set_xlabel("Time from hswa (s)")
        ax.set_ylabel("Counts")

    def plot_raster(self, ax=None):
        pass

    def plot_rippleProb(self, ax=None):
        pass
