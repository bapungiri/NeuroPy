import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.ndimage as filtSig
import scipy.signal as sg
import scipy.stats as stat
from matplotlib.gridspec import GridSpec
from numpy.fft import fft
import matplotlib as mpl

mpl.interactive(False)
# mpl.style.use("seaborn-white")
# import seaborn as sns

mpl.style.use("figPublish")
# sns.set_style("ticks")

from signal_process import filter_sig as filt
from lfpEvent import ripple, hswa
from parsePath import path2files
from behavior import behavior_epochs
from makeChanMap import recinfo

cmap = mpl.cm.get_cmap("jet")

# mpl.use("GtkAgg")


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
    def __init__(self, basePath):

        self.hswa_ripple = hswa_ripple(basePath)
        # self.__ripples = ripple(basePath)


class hswa_ripple(path2files):

    nQuantiles = 10

    def __init__(self, basePath):

        self._myinfo = recinfo(basePath)
        self._ripples = ripple(basePath)
        self._epochs = behavior_epochs(basePath)
        self._swa = hswa(basePath)

    def plot(self):
        """
        calculating the psth for ripple and slow wave oscillation and making n quantiles for plotting 
        """
        # TODO make it general event to event psth which work with time slice instead

        # parameters
        swa_amp_thresh = 0.1
        tbefore = 0.5  # seconds before delta trough
        tafter = 1  # seconds after delta trough

        # pre = self._epochs.pre
        ripplesTime = self._ripples.time
        rippleStart = ripplesTime[:, 0]
        swa_amp = self._swa.amp
        swa_amp_t = self._swa.time

        ind_above_thresh = np.where(swa_amp > swa_amp_thresh)[0]
        swa_amp = swa_amp[ind_above_thresh]
        swa_amp_t = swa_amp_t[ind_above_thresh]

        lfp = self._ripples.best_chan_lfp
        lfp_ripple = filt.filter_ripple(lfp)
        lfp_delta = filt.filter_delta(lfp)
        lfp_t = np.linspace(0, len(lfp) / self._myinfo.lfpSrate, len(lfp))

        # binning with tbefore and tafter delta trough
        _, ripple_co, t_hist = psth(swa_amp_t, rippleStart, [tbefore, tafter])

        quantiles = pd.qcut(swa_amp, self.nQuantiles, labels=False)

        fig = plt.figure(figsize=(6, 10))
        gs = GridSpec(3, 2, figure=fig)

        trial, plt_lfp = 0, 0
        for category in range(self.nQuantiles):
            indx = np.where(quantiles == category)[0]
            ripple_hist = np.sum(ripple_co[indx], axis=0)
            # av_ripple_power = np.sum(ripple_co[indx], axis=0)
            # ripple_hist = filtSig.gaussian_filter1d(ripple_hist, 2)

            nMember_grp = len(indx)
            all_ts = [[] for x in range(nMember_grp)]
            all_y = [[] for x in range(nMember_grp)]
            # ripple_power_arr = []

            for i, ind in enumerate(indx):

                # for ripple number
                ind1 = rippleStart > (swa_amp_t[ind] - tbefore)
                ind2 = rippleStart < (swa_amp_t[ind] + tafter)
                ripple_ts = rippleStart[ind1 & ind2]

                all_ts[i] = ripple_ts - swa_amp_t[ind]
                all_y[i] = (trial + 1) * np.ones(len(ripple_ts))
                trial += 1

                # for ripple power
                # ind1 = np.where(
                #     (lfp_t > (swa_amp_t[ind] - tbefore))
                #     & (lfp_t < (swa_amp_t[ind] + tafter))
                # )
                # # ind2 = lfp_t < (swa_amp_t[ind] + tafter)
                # ripple_power = lfp_ripple[ind1]
                # ripple_power_arr.append(ripple_power)

            # ripple_power_arr = np.asarray(ripple_power_arr)
            # mean_ripple_power_grp = np.mean(ripple_power_arr, axis=0)
            # mean_ripple_power_t = np.linspace(-0.5, 1, len(mean_ripple_power_grp))

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
                    lfp_delta_zsc * 1.2 + y_lfp,
                    color="k",
                )
                ax1.plot(
                    np.linspace(-0.5, 1, len(lfp_zsc)),
                    lfp_ripple_zsc * 0.4 + y_lfp,
                    color=cmap(category / self.nQuantiles),
                    alpha=0.7,
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

            # ax4 = fig.add_subplot(gs[2, 0])
            # ax4.plot(
            #     mean_ripple_power_t,
            #     mean_ripple_power_grp,
            #     color=cmap(category / self.nQuantiles),
            # )

        # plotting aesthetics and labelling
        ax1.spines["left"].set_visible(False)
        ax1.set_yticks([])
        ax1.set_yticklabels([])
        ax1.set_ylim(-1, 50)
        ax1.set_xlabel("Time (s)")

        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("# ripples")

        ax3.spines["left"].set_visible(False)
        ax3.set_yticklabels([])
        ax3.set_xlabel("Time (s)")

        # sns.despine(ax=ax, right=False)
        # plt.close()  # suppressing the output figure
        return fig
