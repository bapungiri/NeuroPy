import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from mathutil import parcorr_mult, getICA_Assembly
import scipy.stats as stats
import matplotlib.gridspec as gridspec


class Replay:
    def __init__(self, obj):
        self.expvar = ExplainedVariance(obj)
        self.bayesian = Bayesian(obj)
        self.assembly = CellAssembly(obj)


class Bayesian:
    def __init__(self, obj):
        self._obj = obj

    def correlation(self, template_time=None, match_time=None, cells=None):
        """Pairwise correlation between template window and matching windows

        Args:
            template_time (array_like, optional): in seconds
            match_time (array_like, optional): in seconds
            cells ([type], optional): cells to calculate the correlation for.

        Returns:
            [array]: returns correlation
        """

        if template_time is None:
            template_time = self._obj.epochs.maze

        if match_time is None:
            match_time = self._obj.epochs.post

        if cells is None:
            unstable_units = self._obj.spikes.stability.unstable
            # stable_units = self._obj.spikes.stability.stable
            # stable_units = list(range(len(spks)))
            stable_units = self._obj.spikes.stability.stable
            quality = np.asarray(self._obj.spikes.info.q)
            pyr = np.where(quality < 5)[0]
            stable_pyr = stable_units[np.isin(stable_units, pyr)]

        spks = self._obj.spikes.times
        print(stable_pyr)

        # spks = [spks[x] for x in stable_units]

        nUnits = len(spks)
        windowSize = self.timeWindow
        spks = [spks[_] for _ in stable_pyr]

        maze_bin = np.arange(maze[0], maze[1], 0.250)
        post_bin = np.arange(post[0], post[1], 0.250)

        pre_spikecount = np.array([np.histogram(x, bins=pre_bin)[0] for x in spks])
        pre_spikecount = [
            pre_spikecount[:, i : i + windowSize]
            for i in range(0, 3 * windowSize, windowSize)
        ]
        maze_spikecount = np.array([np.histogram(x, bins=maze_bin)[0] for x in spks])
        post_spikecount = np.array([np.histogram(x, bins=post_bin)[0] for x in spks])
        post_spikecount = [
            post_spikecount[:, i : i + windowSize]
            for i in range(0, 40 * windowSize, windowSize)
        ]

        # --- pre_corr = np.corrcoef(pre_spikecount)
        pre_corr = [np.corrcoef(pre_spikecount[x]) for x in range(len(pre_spikecount))]
        maze_corr = np.corrcoef(maze_spikecount)
        post_corr = [
            np.corrcoef(post_spikecount[x]) for x in range(len(post_spikecount))
        ]

        # --- selecting only pairwise correlations from different shanks
        shnkId = np.asarray(self._obj.spikes.info.shank)
        shnkId = shnkId[stable_pyr]
        assert len(shnkId) == len(spks)
        cross_shnks = np.nonzero(np.tril(shnkId.reshape(-1, 1) - shnkId.reshape(1, -1)))
        pre_corr = [pre_corr[x][cross_shnks] for x in range(len(pre_spikecount))]
        maze_corr = maze_corr[cross_shnks]
        post_corr = [post_corr[x][cross_shnks] for x in range(len(post_spikecount))]
        print(maze_corr)

        corr_all = []
        for window in range(len(post_corr)):
            nas = np.logical_or(np.isnan(maze_corr), np.isnan(post_corr[window]))
            corr_all.append(np.corrcoef(post_corr[window][~nas], maze_corr[~nas])[0, 1])

        return np.asarray(corr_all)

    def oneD(self):
        pass

    def twoD(self):
        pass


class ExplainedVariance:

    nChans = 16
    binSize = 0.250  # in seconds
    window = 900  # in seconds

    def __init__(self, obj):
        self._obj = obj

    # TODO  smooth version of explained variance
    def compute(self, template=None, match=None, control=None):
        """ Calucate explained variance (EV) and reverse EV
        References:
        1) Kudrimoti 1999
        2) Tastsuno et al. 2007
        """

        if None in [template, match, control]:
            control = self._obj.epochs.pre
            template = self._obj.epochs.maze
            match = self._obj.epochs.post

        # ----- choosing cells ----------------
        spks = self._obj.spikes.times
        stability = self._obj.spikes.stability.info
        stable_pyr = np.where((stability.q < 4) & (stability.stable == 1))[0]
        print(f"Calculating EV for {len(stable_pyr)} stable cells")
        spks = [spks[_] for _ in stable_pyr]

        # ------- windowing the time periods ----------
        window = self.window
        nbins_window = int(window / self.binSize)

        # ---- function to calculate correlation in each window ---------
        def cal_corr(period, windowing=True):
            bin_period = np.arange(period[0], period[1], self.binSize)
            spkcnt = np.array([np.histogram(x, bins=bin_period)[0] for x in spks])

            if windowing:
                dur = np.diff(period)
                nwindow = dur / window
                t = np.arange(period[0], period[1], window)[:-1] + window / 2

                window_spkcnt = [
                    spkcnt[:, i : i + nbins_window]
                    for i in range(0, int(nwindow) * nbins_window, nbins_window)
                ]

                if nwindow % 1 > 0.3:
                    window_spkcnt.append(spkcnt[:, int(nwindow) * nbins_window :])
                    t = np.append(t, round(nwindow % 1, 3) / 2)

                corr = [
                    np.corrcoef(window_spkcnt[x]) for x in range(len(window_spkcnt))
                ]

            else:
                corr = np.corrcoef(spkcnt)
                t = None

            return corr, t

        # ---- correlation for each time period -----------
        control_corr, self.t_control = cal_corr(period=control)
        template_corr, _ = cal_corr(period=template, windowing=False)
        match_corr, self.t_match = cal_corr(period=match)

        # ----- indices for cross shanks correlation -------
        shnkId = np.asarray(self._obj.spikes.info.shank)
        shnkId = shnkId[stable_pyr]
        assert len(shnkId) == len(spks)
        cross_shnks = np.nonzero(np.tril(shnkId.reshape(-1, 1) - shnkId.reshape(1, -1)))

        # --- selecting only pairwise correlations from different shanks -------
        control_corr = [control_corr[x][cross_shnks] for x in range(len(control_corr))]
        template_corr = template_corr[cross_shnks]
        match_corr = [match_corr[x][cross_shnks] for x in range(len(match_corr))]

        parcorr_template_vs_match, rev_corr = parcorr_mult(
            [template_corr], match_corr, control_corr
        )

        ev_template_vs_match = parcorr_template_vs_match ** 2
        rev_corr = rev_corr ** 2

        self.ev = ev_template_vs_match
        self.rev = rev_corr

    def plot(self, ax=None, tstart=0):

        ev_mean = np.mean(self.ev.squeeze(), axis=0)
        ev_std = np.std(self.ev.squeeze(), axis=0)
        rev_mean = np.mean(self.rev.squeeze(), axis=0)
        rev_std = np.std(self.rev.squeeze(), axis=0)

        if ax is None:
            plt.clf()
            fig = plt.figure(1, figsize=(10, 15))
            gs = gridspec.GridSpec(1, 1, figure=fig)
            fig.subplots_adjust(hspace=0.3)
            ax = fig.add_subplot(gs[0])

        t = (self.t_match - tstart) / 3600  # converting to hour

        ax.fill_between(
            t, ev_mean - ev_std, ev_mean + ev_std, color="#7c7979",
        )
        ax.fill_between(
            t, rev_mean - rev_std, rev_mean + rev_std, color="#87d498",
        )

        ax.plot(t, ev_mean, "k")
        ax.plot(t, rev_mean, "#02c59b")
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Explained variance")
        ax.legend(["EV", "REV"])
        ax.text(0.2, 0.28, "POST SD", fontweight="bold")
        # ax.set_xlim([0, 4])


class CellAssembly:
    def __init__(self, obj):
        pass

    def detect(self):
        pass

    def assemblyRusso(self):
        """
            This code implements Russo2017 elife paper for replay detection.
        """

        pass
