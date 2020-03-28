import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.ndimage as filtSig

import scipy.stats as stat
from matplotlib.gridspec import GridSpec
from numpy.fft import fft
import matplotlib as mpl
import time
import scipy.signal as sg


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

    nQuantiles = 10

    def __init__(self, obj):
        self._obj = obj

    def compute(self):
        """
        calculating the psth for ripple and slow wave oscillation and making n quantiles for plotting 
        """

        # parameters
        swa_amp_thresh = 0.1
        tbefore = 0.5  # seconds before delta trough
        tafter = 1  # seconds after delta trough

        ripplesTime = self._obj.ripple.time
        rippleStart = ripplesTime[:, 0]
        swa_amp = self._obj.swa.amp
        swa_amp_t = self._obj.swa.time

        ind_above_thresh = np.where(swa_amp > swa_amp_thresh)[0]
        swa_amp = swa_amp[ind_above_thresh]
        swa_amp_t = swa_amp_t[ind_above_thresh]

        lfp, _ = self._obj.ripple.best_chan_lfp
        lfp_ripple = filt.filter_ripple(lfp)
        # lfp_delta = filt.filter_delta(lfp)

        analytic_signal = sg.hilbert(lfp_ripple)
        amplitude_envelope = np.abs(analytic_signal)

        # binning with tbefore and tafter delta trough
        _, ripple_co, t_hist = psth(swa_amp_t, rippleStart, [tbefore, tafter])

        quantiles = pd.qcut(swa_amp, self.nQuantiles, labels=False)

        ripple_psth = pd.DataFrame(t_hist[:-1], columns=["time"])
        ripple_power = pd.DataFrame(
            np.linspace(-tbefore, tafter, (tafter + tbefore) * 1250), columns=["time"]
        )

        for category in range(self.nQuantiles):
            indx = np.where(quantiles == category)[0]
            ripple_hist = np.sum(ripple_co[indx], axis=0)
            # av_ripple_power = np.sum(ripple_co[indx], axis=0)
            # ripple_hist = filtSig.gaussian_filter1d(ripple_hist, 2)
            # ripple_psth.append(ripple_hist)
            ripple_psth[category] = ripple_hist

            ripple_power_arr = []
            for i, ind in enumerate(indx):

                if self._obj.trange.any():
                    frame = int(self._obj.trange[0] * 1250)
                    swa_frame = int(swa_amp_t[ind] * 1250) - frame
                else:
                    swa_frame = int(swa_amp_t[ind] * 1250)

                # making sure swa_frame are inside array indices of trange lfp
                if swa_frame > 625 and swa_frame + 1250 < len(lfp):
                    ripp_pow = amplitude_envelope[swa_frame - 625 : swa_frame + 1250]
                    ripple_power_arr.append(ripp_pow)

            nMember_grp = len(indx)
            ripple_power_arr = np.asarray(ripple_power_arr)
            mean_ripple_power_grp = np.mean(ripple_power_arr, axis=0)
            ripple_std = np.std(ripple_power_arr, axis=0) / np.sqrt(nMember_grp)

            ripple_power[category] = mean_ripple_power_grp
            ripple_power["std" + str(category)] = ripple_std
        self.ripple_psth = ripple_psth
        self.ripple_power = ripple_power

    # plotting methods

    def plot_ripplePower(self, ax=None):
        cmap = mpl.cm.get_cmap("viridis")
        colmap = [cmap(x) for x in np.linspace(0, 1, self.nQuantiles)]
        self.ripple_power.plot(
            x="time",
            y=[_ for _ in range(self.nQuantiles)],
            ax=ax,
            legend=False,
            colormap=cmap,
        )
        for _ in range(self.nQuantiles):

            ax.fill_between(
                self.ripple_power["time"],
                self.ripple_power[_] + self.ripple_power["std" + str(_)],
                self.ripple_power[_] - self.ripple_power["std" + str(_)],
                color=colmap[_],
                alpha=0.4,
            )

    def plot_raster(self, ax=None):
        pass

    def plot_rippleProb(self, ax=None):

        self.ripple_psth.plot(
            x="time", y=[_ for _ in range(self.nQuantiles)], ax=ax, legend=False
        )
