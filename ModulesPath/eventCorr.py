import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.ndimage as filtSig
import scipy.signal as sg
import scipy.stats as stat
from matplotlib.gridspec import GridSpec
from numpy.fft import fft

from parsePath import name2path
from signal_process import filter_sig as filt

cmap = matplotlib.cm.get_cmap("jet")


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


class hswa_ripple(name2path):

    nQuantiles = 10

    def __init__(self, basePath):
        super().__init__(basePath)

    def plot_psth_hswa_ripple(self):
        """
        calculating the psth for ripple and slow wave oscillation and making n quantiles for plotting 
        """
        # TODO make it general event to event psth which work with time slice instead

        # parameters
        swa_amp_thresh = 0.1

        epochs = np.load(str(self.filePrefix) + "_epochs.npy", allow_pickle=True).item()
        pre = epochs["PRE"]  # in seconds
        maze = epochs["MAZE"]  # in seconds
        post = epochs["POST"]  # in seconds
        ripples = np.load(self.f_ripple_evt, allow_pickle=True).item()
        ripplesTime = ripples["timestamps"]
        rippleStart = ripplesTime[:, 0]

        delta_evts = np.load(self.f_slow_wave, allow_pickle=True).item()
        swa_amp = delta_evts["delta_amp"]
        swa_amp_t = delta_evts["delta_t"]

        ind_above_thresh = np.where(swa_amp > swa_amp_thresh)[0]
        swa_amp = swa_amp[ind_above_thresh]
        swa_amp_t = swa_amp_t[ind_above_thresh]

        lfp = np.load(self.f_ripplelfp, allow_pickle=True).item()
        lfp = lfp["BestChan"]
        lfp_ripple = filt.filter_ripple(lfp)
        lfp_delta = filt.filter_delta(lfp)
        lfp_t = np.linspace(0, len(lfp) / self.lfpsRate, len(lfp))

        # binning with 500ms before and 1 sec after
        ripple_hist, ripple_co, t_hist = psth(swa_amp_t, rippleStart, [0.5, 1])

        quantiles = pd.qcut(swa_amp, self.nQuantiles, labels=False)

        fig = plt.figure(figsize=(6, 10))
        gs = GridSpec(3, 3, figure=fig)

        trial = 0
        plt_lfp = 0

        for category in range(self.nQuantiles):
            indx = np.where(quantiles == category)[0]
            ripple_hist = np.sum(ripple_co[indx], axis=0)
            ripple_hist = filtSig.gaussian_filter1d(ripple_hist, 2)
            all_ts = [[] for x in range(len(indx))]
            all_y = [[] for x in range(len(indx))]
            for i, ind in enumerate(indx):

                ripple_around = np.where(
                    (rippleStart > swa_amp_t[ind] - 0.5)
                    & (rippleStart < swa_amp_t[ind] + 1)
                )[0]
                ripple_ts = rippleStart[ripple_around]

                # trial_on = raster_data.time_ranges[0, trial]
                # trial_off = raster_data.time_ranges[1, trial]
                # ind1 = ts >= trial_on
                # ind2 = ts < trial_off
                # trial_ts = ts[ind1 & ind2]
                all_ts[i] = ripple_ts - swa_amp_t[ind]
                all_y[i] = (trial + 1) * np.ones(len(ripple_ts))
                trial += 1

            # random selection of slow wave within the quantile
            rand_ind = np.random.choice(indx, 3, replace=False)

            ax1 = fig.add_subplot(gs[:, 1])
            for ind in rand_ind:
                print(swa_amp[ind])
                ind_theta = np.where(
                    (lfp_t > swa_amp_t[ind] - 0.5) & (lfp_t < swa_amp_t[ind] + 1)
                )[0]

                lfp_zsc = stat.zscore(lfp[ind_theta])
                lfp_ripple_zsc = stat.zscore(lfp_ripple[ind_theta])
                lfp_delta_zsc = stat.zscore(lfp_delta[ind_theta])
                y_lfp = (plt_lfp + 2) * np.ones(len(lfp_zsc))
                ax1.plot(
                    np.linspace(-0.5, 1, len(lfp_zsc)),
                    lfp_ripple_zsc * 0.4 + y_lfp,
                    color=cmap(category / self.nQuantiles),
                )
                ax1.plot(
                    np.linspace(-0.5, 1, len(lfp_zsc)),
                    lfp_delta_zsc * 1.2 + y_lfp,
                    color="k",
                )
                plt_lfp += 3

            ax2 = fig.add_subplot(gs[0, 0])
            ax2.plot(t_hist[:-1], ripple_hist, color=cmap(category / self.nQuantiles))

            ax3 = fig.add_subplot(gs[1, 0])
            all_x = np.concatenate(all_ts)
            all_y = np.concatenate(all_y)
            ax3.plot(
                all_x, all_y, ".", markersize=1, color=cmap(category / self.nQuantiles)
            )
