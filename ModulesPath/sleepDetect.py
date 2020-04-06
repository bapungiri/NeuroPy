import numpy as np
import matplotlib.pyplot as plt
from SpectralAnalysis import bestThetaChannel
import pandas as pd
import numpy as np
import scipy.signal as sg
import scipy.stats as stats
from hmmlearn.hmm import GaussianHMM
import scipy.ndimage as filtSig
import os
import pathlib as Path
from signal_process import spectrogramBands


def genepoch(start, end):

    if start[0] > end[0]:
        end = end[1:]

    if start[-1] > end[-1]:
        start = start[:-1]

    firstPass = np.vstack((start, end)).T

    # ===== merging close ripples
    minInterSamples = 20
    secondPass = []
    state = firstPass[0]
    for i in range(1, len(firstPass)):
        if firstPass[i, 0] - state[1] < minInterSamples:
            # Merging states
            state = [state[0], firstPass[i, 1]]
        else:
            secondPass.append(state)
            state = firstPass[i]

    secondPass.append(state)
    secondPass = np.asarray(secondPass)

    state_duration = np.diff(secondPass, axis=1)

    # delete very short ripples
    minstateDuration = 90
    shortRipples = np.where(state_duration < minstateDuration)[0]
    thirdPass = np.delete(secondPass, shortRipples, 0)

    return thirdPass


def hmmfit1d(Data):
    # hmm states on 1d data and returns labels with highest mean = highest label
    Data = (np.asarray(Data)).reshape(-1, 1)
    model = GaussianHMM(n_components=2, n_iter=100).fit(Data)
    hidden_states = model.predict(Data)
    mus = np.squeeze(model.means_)
    sigmas = np.squeeze(np.sqrt(model.covars_))
    transmat = np.array(model.transmat_)

    idx = np.argsort(mus)
    mus = mus[idx]
    sigmas = sigmas[idx]
    transmat = transmat[idx, :][:, idx]

    state_dict = {}
    states = [i for i in range(4)]
    for i in idx:
        state_dict[idx[i]] = states[i]

    relabeled_states = np.asarray([state_dict[h] for h in hidden_states])
    relabeled_states[:2] = [0, 0]
    relabeled_states[-2:] = [0, 0]

    state_diff = np.diff(relabeled_states)
    start = np.where(state_diff == 1)[0]
    stop = np.where(state_diff == -1)[0]

    for s, e in zip(start, stop):
        if e - s < 50:
            relabeled_states[s + 1 : e] = 0
    # print(start_ripple.shape, stop_ripple.shape)
    # states = np.concatenate((start_ripple, stop_ripple), axis=1)

    # relabeled_states = hidden_states
    return relabeled_states


class SleepScore:

    window = 1  # seconds
    overlap = 0.2  # seconds

    def __init__(self, obj):
        self._obj = obj

    def detect(self):

        # a = np.array([self._obj.epochs.pre[0], self._obj.epochs.post[1]])
        emg = self._emgfromlfp(fromfile=0)
        self.params_pre, self.sxx_pre = self._getparams(self._obj.epochs.pre, emg)
        self.params_maze, self.sxx_maze = self._getparams(self._obj.epochs.maze, emg)
        self.params_post, self.sxx_post = self._getparams(self._obj.epochs.post, emg)

        # self.maze_params, self.maze_spect = self._getparams(self._obj.epochs.maze)
        # self.maze_params, self.sxx = self._getparams(self._obj.epochs.maze)
        # self.post_params, self.sxx = self._getparams(self._obj.epochs.post)

    @staticmethod
    def _label2states(theta_delta, delta_l, emg_l):

        state = np.zeros(len(theta_delta))
        for i, (ratio, delta, emg) in enumerate(zip(theta_delta, delta_l, emg_l)):

            # if ratio == 1 and emg == 1:
            #     state[i] = 4
            # elif ratio == 0 and emg == 1 and delta == 0:
            #     state[i] = 3
            # elif ratio == 1 and emg == 0:
            #     state[i] = 2
            # elif ratio == 0 and emg == 1 and delta == 1:
            #     state[i] = 1
            # elif ratio == 0 and emg == 0:
            #     state[i] = 1
            if ratio == 1 and emg == 1:
                state[i] = 4
            elif ratio == 0 and emg == 1:
                state[i] = 3
            elif ratio == 1 and emg == 0:
                state[i] = 2
            elif ratio == 0 and emg == 0:
                state[i] = 1
            # elif ratio == 0 and emg == 1 and delta == 1:
            #     state[i] = 1
            # elif ratio == 0 and emg == 0:
            #     state[i] = 1

        return state

    def _getparams(self, interval, emg):
        sRate = self._obj.recinfo.lfpSrate

        lfp = np.load(self._obj.files.thetalfp)
        # ripplelfp = np.load(self._obj.files.ripplelfp).item()["BestChan"]

        lfp = stats.zscore(lfp)
        bands = spectrogramBands(
            lfp, window=self.window * sRate, overlap=self.overlap * sRate
        )
        time = bands.time

        ind = np.where((time > interval[0]) & (time < interval[1]))[0]
        delta = bands.delta[ind]
        theta = bands.theta[ind]
        spindle = bands.spindle[ind]
        gamma = bands.gamma[ind]
        ripple = bands.ripple[ind]
        theta_delta_ratio = stats.zscore(theta / delta)
        theta_delta_label = hmmfit1d(theta_delta_ratio)
        delta_label = hmmfit1d(delta)
        sxx = stats.zscore(bands.sxx[:, ind], axis=None)

        emg_label = hmmfit1d(emg)
        emg = emg[ind]
        emg_label = emg_label[ind]

        states = self._label2states(theta_delta_label, delta_label, emg_label)

        data = pd.DataFrame(
            {
                "delta": stats.zscore(delta),
                "theta": stats.zscore(theta),
                "spindle": stats.zscore(spindle),
                "gamma": stats.zscore(gamma),
                "ripple": stats.zscore(ripple),
                "theta_delta_ratio": theta_delta_ratio,
                "emg": emg,
                "state": states,
            }
        )
        # data_label = pd.DataFrame({"theta_delta": theta_delta_label, "emg": emg_label})

        return data, sxx

    def _emgfromlfp(self, fromfile=0):

        if fromfile:
            emg_lfp = np.load(self._obj.files.corr_emg)

        else:
            highfreq = 600
            lowfreq = 300
            sRate = self._obj.recinfo.lfpSrate
            nChans = self._obj.recinfo.nChans
            nyq = 0.5 * sRate
            nShanks = self._obj.recinfo.nShanks
            # channels = self._obj.recinfo.channels
            changroup = self._obj.recinfo.channelgroups
            changroup = changroup[:nShanks]
            badchans = self._obj.recinfo.badchans
            chan_top = [
                np.setdiff1d(channels, badchans, assume_unique=True)[0]
                for channels in changroup
            ]
            # chan_middle = [
            #     np.setdiff1d(channels, badchans, assume_unique=True)[8]
            #     for channels in changroup
            # ]
            chan_bottom = [
                np.setdiff1d(channels, badchans, assume_unique=True)[-1]
                for channels in changroup
            ]
            chan_map_select = [chan_top[0], chan_top[-1]]
            # chan_map_select = np.union1d(chan_top, chan_bottom)
            # chan_map_select = np.setdiff1d(channels, badchans, assume_unique=True)

            # filtering for high frequency band
            eegdata = np.memmap(self._obj.recfiles.eegfile, dtype="int16", mode="r")
            eegdata = np.memmap.reshape(eegdata, (int(len(eegdata) / nChans), nChans))
            lfp_req = eegdata[:, chan_map_select]
            b, a = sg.butter(3, [lowfreq / nyq, highfreq / nyq], btype="bandpass")
            yf = sg.filtfilt(b, a, lfp_req, axis=0)

            # windowing signal
            frames = np.arange(
                0,
                len(eegdata) - self.window * sRate,
                (self.window - self.overlap) * sRate,
            )

            emg_lfp = []
            for start in frames:

                start_frame = int(start)
                end_frame = start_frame + self.window * sRate
                temp = (yf[start_frame:end_frame, :]).T
                corr_chan = np.corrcoef(temp)
                corr_all = corr_chan[np.tril_indices(len(corr_chan), k=-1)]
                # corr_all = temp[0, :]
                emg_lfp.append(np.sum(corr_all))

            np.save(self._obj.files.corr_emg, emg_lfp)
            print(corr_all.shape)

        emg_smooth = filtSig.gaussian_filter1d(emg_lfp, 10, axis=0)

        return emg_smooth
