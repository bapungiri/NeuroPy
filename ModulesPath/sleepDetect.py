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

    # for s, e in zip(start, stop):
    #     if e - s < 50:
    #         relabeled_states[s + 1 : e] = 0
    # print(start_ripple.shape, stop_ripple.shape)
    # states = np.concatenate((start_ripple, stop_ripple), axis=1)

    # relabeled_states = hidden_states
    return relabeled_states


class SleepScore:

    window = 1  # seconds
    overlap = 0.2  # seconds

    def __init__(self, obj):
        self._obj = obj
        self.params = pd.read_pickle(self._obj.files.stateparams)
        self.states = pd.read_pickle(self._obj.files.states)

    def detect(self):

        # a = np.array([self._obj.epochs.pre[0], self._obj.epochs.post[1]])
        emg = self._emgfromlfp(fromfile=0)
        params_pre, sxx_pre, states_pre = self._getparams(self._obj.epochs.pre, emg)
        params_maze, sxx_maze, states_maze = self._getparams(self._obj.epochs.maze, emg)
        params_post, sxx_post, states_post = self._getparams(self._obj.epochs.post, emg)

        df = [params_pre, params_maze, params_post]
        states = [states_pre, states_maze, states_post]

        self.params = pd.concat(df, ignore_index=True)
        self.sxx = np.concatenate((sxx_pre, sxx_maze, sxx_post), axis=1)
        self.states = pd.concat(states, ignore_index=True)

        self.params.to_pickle(self._obj.files.stateparams)
        self.states.to_pickle(self._obj.files.states)

    @staticmethod
    def _label2states(theta_delta, delta_l, emg_l):

        state = np.zeros(len(theta_delta))
        for i, (ratio, delta, emg) in enumerate(zip(theta_delta, delta_l, emg_l)):

            if ratio == 1 and emg == 1:
                state[i] = 4
            elif ratio == 0 and emg == 1:
                state[i] = 3
            elif ratio == 1 and emg == 0:
                state[i] = 2
            elif ratio == 0 and emg == 0:
                state[i] = 1

        return state

    @staticmethod
    def _states2time(label):

        states = np.unique(label)

        all_states = []
        for state in states:

            binary = np.where(label == state, 1, 0)
            binary = np.concatenate(([0], binary, [0]))
            binary_change = np.diff(binary)

            start = np.where(binary_change == 1)[0]
            end = np.where(binary_change == -1)[0]
            start = start[:-1]
            end = end[:-1]
            # duration = end - start
            stateid = state * np.ones(len(start))
            firstPass = np.vstack((start, end, stateid)).T

            all_states.append(firstPass)

        all_states = np.concatenate(all_states)

        return all_states

    @staticmethod
    def _removetransient(statetime):

        duration = statetime.duration
        start = statetime.start
        end = statetime.end
        state = statetime.state

        arr = np.zeros((len(start), 4))
        arr[:, 0] = start
        arr[:, 1] = end
        arr[:, 2] = duration
        arr[:, 3] = state

        srt_ind = np.argsort(arr[:, 0])
        arr = arr[srt_ind, :]

        ind = 1
        while ind < len(arr) - 1:
            if (arr[ind, 2] < 50) and (arr[ind - 1, 3] == arr[ind + 1, 3]):
                arr[ind - 1, :] = np.array(
                    [
                        arr[ind - 1, 0],
                        arr[ind + 1, 1],
                        arr[ind + 1, 1] - arr[ind - 1, 0],
                        arr[ind - 1, 3],
                    ]
                )
                arr = np.delete(arr, [ind, ind + 1], 0)
            else:
                ind += 1
        # new_state = []
        # for i in range(0, len(start) - 2):
        #     if (end[i] - start[i + 2]) < 5 and (state[i] == state[i + 2]):
        #         new_state.append(
        #             [start[i], end[i + 2], end[i + 2] - start[i], state[i]]
        #         )
        #         i = i + 3
        #     else:
        #         new_state.append([start[i], end[i], duration[i], state[i]])

        # new_state = np.asarray(new_state)

        statetime = pd.DataFrame(
            {
                "start": arr[:, 0],
                "end": arr[:, 1],
                "duration": arr[:, 2],
                "state": arr[:, 3],
            }
        )

        return statetime

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
        t = time[ind]
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
        statetime = (self._states2time(states)).astype(int)

        data = pd.DataFrame(
            {
                "time": t,
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

        statetime = pd.DataFrame(
            {
                "start": t[statetime[:, 0]],
                "end": t[statetime[:, 1]],
                "duration": t[statetime[:, 1]] - t[statetime[:, 0]],
                "state": statetime[:, 2],
            }
        )

        statetime_new = self._removetransient(statetime)

        # data_label = pd.DataFrame({"theta_delta": theta_delta_label, "emg": emg_label})

        return data, sxx, statetime_new

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
            # print(changroup)
            badchans = self._obj.recinfo.badchans

            if len(badchans) > 30:

                changroup = changroup[:4]
            else:
                changroup = changroup[:nShanks]

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
            chan_map_select = [chan_top[0], chan_bottom[-1]]
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

        emg_smooth = filtSig.gaussian_filter1d(emg_lfp, 20, axis=0)

        return emg_smooth
