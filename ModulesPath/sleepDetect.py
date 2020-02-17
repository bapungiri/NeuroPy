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
from parsePath import name2path


def hmmfit1d(Data):
    # hmm states on 1d data and returns labels with highest mean = highest label
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


class SleepScore(name2path):
    lfpsampleRate = 1250
    nyq = 0.5 * lfpsampleRate

    nshanks = 8
    window = 1  # seconds
    slideby = 0.5  # seconds

    def __init__(self, basePath):
        super().__init__(basePath)

        thetafile = self.filePrefix + "_BestThetaChan.npy"
        self.thetaData = np.load(thetafile, allow_pickle=True)

        self.bad_chans = np.load(self.filePrefix + "_badChans.npy", allow_pickle=True)
        basics = np.load(self.filePrefix + "_basics.npy", allow_pickle=True)
        self.chan_map = basics.item().get("channels")
        self.nChans = basics.item().get("nChans")
        eegdata = np.memmap(self.filename, dtype="int16", mode="r")
        self.eegdata = np.memmap.reshape(
            eegdata, (int(len(eegdata) / self.nChans), self.nChans)
        )

    def spect(self, data):
        f, t, sxx = sg.spectrogram(
            self.thetaData,
            fs=self.lfpsampleRate,
            nperseg=self.window * self.lfpsampleRate,
            noverlap=self.slideby * self.lfpsampleRate,
        )

        return f, t, sxx

    def deltaStates(self):

        f, t, sxx = self.spect(self.thetaData)

        delta_ind = np.where((f > 1) & (f < 4))[0]  # delta band 0-4 Hz and
        theta_ind = np.where((f > 4) & (f < 11))[0]  # theta band 0-4 Hz and

        theta_sxx = np.mean(sxx[theta_ind, :], axis=0)

        delta_sxx = np.mean(sxx[delta_ind, :], axis=0)
        delta_smooth = filtSig.gaussian_filter1d(delta_sxx, 10, axis=0)
        delta_smooth = np.reshape(delta_smooth, [len(delta_smooth), 1])

        states = hmmfit1d(delta_smooth)

        self.delta_states = np.array(states)
        self.delta = stats.zscore(delta_smooth)
        self.deltaraw = stats.zscore(delta_sxx)

        sleep_stages = []
        sleep_state = np.diff(states)
        state_start = np.where(sleep_state == 1)[0]
        state_end = np.where(sleep_state == -1)[0]

        state_prune = genepoch(state_start, state_end)
        self.sec = state_prune
        print(len(state_prune))
        self.sws_time = np.vstack((t[state_prune[:, 0]], t[state_prune[:, 1]])).T
        self.delta_t = t
        # sleep_stages.extend(firstPass)

        # sleep_stages = np.asarray(sleep_stages)
        # self.sleep_stages = sleep_stages[sleep_stages[:, 0].argsort()]

    def thetaStates(self):
        f, _, sxx = self.spect(self.thetaData)

        theta_ind = np.where((f > 1) & (f < 4))[0]  # theta band 0-4 Hz and 12-15 Hz

        theta_sxx = np.mean(sxx[theta_ind, :], axis=0)
        theta_smooth = filtSig.gaussian_filter1d(theta_sxx, 10, axis=0)
        theta_smooth = np.reshape(theta_smooth, [len(theta_smooth), 1])

        states = hmmfit1d(theta_smooth)
        self.theta_states = np.array(states)
        self.theta = stats.zscore(theta_smooth)
        self.thetaraw = stats.zscore(theta_sxx)

    def emgfromlfp(self):
        highfreq = 300
        lowfreq = 275
        nyq = self.nyq
        # chan_per_shank = nChans / self.nshanks
        # TODO top channel selection
        chan_multiple = int((self.nChans / self.nshanks) / self.nshanks)
        chan_top = self.chan_map[:: self.nshanks * chan_multiple]  # select top channels
        chan_bottom = self.chan_map[
            self.nshanks * chan_multiple - 1 :: self.nshanks * chan_multiple
        ]  # select bottom channels
        chan_map_select = np.union1d(chan_top, chan_bottom)

        # filtering for high frequency band
        chan_good = np.setdiff1d(chan_map_select, self.bad_chans)
        lfp_req = self.eegdata[:, chan_good]
        b, a = sg.butter(3, [lowfreq / nyq, highfreq / nyq], btype="bandpass")
        yf = sg.filtfilt(b, a, lfp_req, axis=0)

        # windowing signal
        frames = np.arange(
            0,
            len(self.eegdata) - self.window * self.lfpsampleRate,
            self.slideby * self.lfpsampleRate,
        )

        self.emg_lfp = []
        for i in range(len(frames)):

            start_frame = i * self.slideby * self.lfpsampleRate
            end_frame = start_frame + self.window * self.lfpsampleRate
            temp = (yf[start_frame:end_frame, :]).T
            corr_chan = np.corrcoef(temp)
            corr_all = corr_chan[np.tril_indices(len(corr_chan), k=-1)]
            self.emg_lfp.append(np.sum(corr_all))

    def thetaDeltaratio(self):
        f, _, sxx = self.spect(self.thetaData)

        theta_ind = np.where((f > 5) & (f < 10))[0]
        delta_ind = np.where((f < 4) | ((f > 12) & (f < 16)))[
            0
        ]  # delta band 0-4 Hz and 12-15 Hz
        gamma_ind = np.where((f > 50) & (f < 250))[0]  # delta band 0-4 Hz and 12-15 Hz

        theta_sxx = np.mean(sxx[theta_ind, :], axis=0)
        delta_sxx = np.mean(sxx[delta_ind, :], axis=0)
        gamma_sxx = np.mean(sxx[gamma_ind, :], axis=0)

        theta_delta_ratio = stats.zscore(delta_sxx / theta_sxx)
        theta_gamma_ratio = stats.zscore(theta_sxx / gamma_sxx)
        theta_delta_ratio = np.reshape(theta_delta_ratio, [len(theta_delta_ratio), 1])

        theta_delta_smooth = filtSig.gaussian_filter1d(theta_delta_ratio, 3, axis=0)
        feature_comb = np.hstack(
            (theta_delta_ratio, np.reshape(self.emg_lfp, (len(self.emg_lfp), 1)))
        )

        model = GaussianHMM(n_components=4, n_iter=100).fit(feature_comb)
        hidden_states = model.predict(feature_comb)

        relabeled_states = hidden_states
        # theta_ratio_dist, bin_edges = np.histogram(theta_delta_ratio,bins=100)

        # plt.plot(theta_gamma_ratio,theta_delta_ratio,'.')
        # plt.plot(freq[:N//2], (2/N)*fftSig[:N//2])

        relabeled_states = np.array(relabeled_states)
        # sleep states labelling

        sleep_stages = []
        for i in range(4):

            sleep_state = np.where(relabeled_states == i, 1, 0)
            sleep_state = np.diff(sleep_state)
            state_start = np.where(sleep_state == 1)[0]
            state_end = np.where(sleep_state == -1)[0]
            state_label = i * np.ones(len(state_start))
            firstPass = np.vstack((t[state_start], t[state_end], state_label)).T
            sleep_stages.extend(firstPass)

        sleep_stages = np.asarray(sleep_stages)
        sleep_stages = sleep_stages[sleep_stages[:, 0].argsort()]

        # np.save(basePath + sessionName + "_behavior.npy", sleep_stages)

        arr_start = np.argwhere(f > 25)[0]
        sxx2 = sxx[: arr_start[0]][:]
        # sxx2 = np.flipud(sxx2)

        # plt.clf()

        # plt.imshow(
        #     sxx2,
        #     cmap="YlGn",
        #     aspect="auto",
        #     extent=[0, len(t), 0, 25.0],
        #     origin="lower",
        #     vmin=-500,
        #     vmax=140000,
        #     interpolation="mitchell",
        # )
        # # plt.pcolormesh(t / 3600, f, sxx, cmap="copper", vmax=30)

        # # plt.plot(theta_delta_ratio)
        # plt.plot((theta_delta_smooth + 5) * 2, "r", linewidth=2)
        # plt.plot(relabeled_states + 4, color="#3fa8d5", linewidth=3)
        # plt.plot(emg_lfp * 50)
        # # plt.plot(hidden_states, "r")

