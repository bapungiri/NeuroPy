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

    relabeled_states = [state_dict[h] for h in hidden_states]
    # relabeled_states = hidden_states
    return relabeled_states


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


class SleepScore:

    window = 1  # seconds
    slideby = 0.5  # seconds

    def __init__(self, obj):
        self._obj = obj

    def detect(self):
        nChans = self._obj.recinfo.nChans
        eegdata = np.memmap(self._obj.recfiles.eegfile, dtype="int16", mode="r")
        eegdata = np.memmap.reshape(eegdata, (int(len(eegdata) / nChans), nChans))

        a = np.array([self._obj.epochs.pre[0], self._obj.epochs.maze[1]])
        self.pre_params, self.sxx = self._getparams(a)
        # self.maze_params, self.maze_spect = self._getparams(self._obj.epochs.maze)
        # self.maze_params, self.sxx = self._getparams(self._obj.epochs.maze)
        # self.post_params, self.sxx = self._getparams(self._obj.epochs.post)

    def _getparams(self, interval):
        sRate = self._obj.recinfo.lfpSrate
        nChans = self._obj.recinfo.nChans
        frames = (interval * sRate).astype(int)

        lfp = np.load(self._obj.files.thetalfp, allow_pickle=True)
        lfp = stats.zscore(lfp)
        lfp = lfp[frames[0] : frames[1]]

        eegdata = np.memmap(self._obj.recfiles.eegfile, dtype="int16", mode="r")
        eegdata = np.memmap.reshape(eegdata, (int(len(eegdata) / nChans), nChans))
        # eegdata = eegdata[frames[0] : frames[1], :]

        theta, sxx_thetachan = self._theta(lfp)
        theta_label = hmmfit1d(theta)

        delta, _ = self._delta(lfp)
        delta_label = hmmfit1d(delta)

        emg = self._emgfromlfp(eegdata)
        emg_label = hmmfit1d(emg)

        data = pd.DataFrame({"theta": theta, "delta": delta, "emg": emg})

        return data, sxx_thetachan

    def _spect(self, lfp):

        sampfreq = self._obj.recinfo.lfpSrate
        nperseg = self.window * sampfreq
        noverlap = self.slideby * sampfreq

        f, t, sxx = sg.spectrogram(lfp, fs=sampfreq, nperseg=nperseg, noverlap=noverlap)

        return f, t, sxx

    def _delta(self, lfp):
        # lfp = stats.zscore(lfp)
        f, _, sxx = self._spect(lfp)

        delta_ind = np.where(((f > 0.5) & (f < 4)) | ((f > 12) & (f < 15)))[0]
        # delta_ind = np.where(((f > 0.5) & (f < 16)))[0]

        delta_sxx = np.mean(sxx[delta_ind, :], axis=0)
        delta_smooth = filtSig.gaussian_filter1d(delta_sxx, 10, axis=0)
        # delta_smooth = np.reshape(delta_smooth, [len(delta_smooth), 1])

        return delta_smooth, sxx

    def _theta(self, lfp):
        # lfp = stats.zscore(lfp)
        f, _, sxx = self._spect(lfp)

        theta_ind = np.where((f > 5) & (f < 11))[0]  # theta band 0-4 Hz and 12-15 Hz

        theta_sxx = np.mean(sxx[theta_ind, :], axis=0)
        theta_smooth = filtSig.gaussian_filter1d(theta_sxx, 10, axis=0)
        # theta_smooth = np.reshape(theta_smooth, [len(theta_smooth), 1])

        return theta_smooth, sxx

    def _emgfromlfp(self, eegdata):
        highfreq = 600
        lowfreq = 300
        sRate = self._obj.recinfo.lfpSrate
        nyq = 0.5 * sRate
        nShanks = self._obj.recinfo.nShanks
        changroup = self._obj.recinfo.channelgroups
        changroup = changroup[:nShanks]
        badchans = self._obj.recinfo.badchans
        chan_top = [
            np.setdiff1d(channels, badchans, assume_unique=True)[0]
            for channels in changroup
        ]
        chan_bottom = [
            np.setdiff1d(channels, badchans, assume_unique=True)[-1]
            for channels in changroup
        ]
        chan_map_select = np.union1d(chan_top, chan_bottom)

        # filtering for high frequency band
        lfp_req = eegdata[:, chan_map_select]
        b, a = sg.butter(3, [lowfreq / nyq, highfreq / nyq], btype="bandpass")
        yf = sg.filtfilt(b, a, lfp_req, axis=0)

        # windowing signal
        frames = np.arange(0, len(eegdata) - self.window * sRate, self.slideby * sRate)

        emg_lfp = []
        for i in range(len(frames)):

            start_frame = int(i * self.slideby * sRate)
            end_frame = start_frame + self.window * sRate
            temp = (yf[start_frame:end_frame, :]).T
            corr_chan = np.corrcoef(temp)
            corr_all = corr_chan[np.tril_indices(len(corr_chan), k=-1)]
            emg_lfp.append(np.mean(corr_all))

        emg_smooth = filtSig.gaussian_filter1d(emg_lfp, 10, axis=0)

        return emg_smooth

    # def thetaDeltaratio(self):
    #     f, _, sxx = self.spect(self.thetaData)

    #     theta_ind = np.where((f > 5) & (f < 10))[0]
    #     delta_ind = np.where((f < 4) | ((f > 12) & (f < 16)))[
    #         0
    #     ]  # delta band 0-4 Hz and 12-15 Hz
    #     gamma_ind = np.where((f > 50) & (f < 250))[0]  # delta band 0-4 Hz and 12-15 Hz

    #     theta_sxx = np.mean(sxx[theta_ind, :], axis=0)
    #     delta_sxx = np.mean(sxx[delta_ind, :], axis=0)
    #     gamma_sxx = np.mean(sxx[gamma_ind, :], axis=0)

    #     theta_delta_ratio = stats.zscore(delta_sxx / theta_sxx)
    #     theta_gamma_ratio = stats.zscore(theta_sxx / gamma_sxx)
    #     theta_delta_ratio = np.reshape(theta_delta_ratio, [len(theta_delta_ratio), 1])

    #     theta_delta_smooth = filtSig.gaussian_filter1d(theta_delta_ratio, 3, axis=0)
    #     feature_comb = np.hstack(
    #         (theta_delta_ratio, np.reshape(self.emg_lfp, (len(self.emg_lfp), 1)))
    #     )

    #     model = GaussianHMM(n_components=4, n_iter=100).fit(feature_comb)
    #     hidden_states = model.predict(feature_comb)

    #     relabeled_states = hidden_states
    #     # theta_ratio_dist, bin_edges = np.histogram(theta_delta_ratio,bins=100)

    #     # plt.plot(theta_gamma_ratio,theta_delta_ratio,'.')
    #     # plt.plot(freq[:N//2], (2/N)*fftSig[:N//2])

    #     relabeled_states = np.array(relabeled_states)
    #     # sleep states labelling

    #     sleep_stages = []
    #     for i in range(4):

    #         sleep_state = np.where(relabeled_states == i, 1, 0)
    #         sleep_state = np.diff(sleep_state)
    #         state_start = np.where(sleep_state == 1)[0]
    #         state_end = np.where(sleep_state == -1)[0]
    #         state_label = i * np.ones(len(state_start))
    #         firstPass = np.vstack((t[state_start], t[state_end], state_label)).T
    #         sleep_stages.extend(firstPass)

    #     sleep_stages = np.asarray(sleep_stages)
    #     sleep_stages = sleep_stages[sleep_stages[:, 0].argsort()]

    #     # np.save(basePath + sessionName + "_behavior.npy", sleep_stages)

    #     arr_start = np.argwhere(f > 25)[0]
    #     sxx2 = sxx[: arr_start[0]][:]
