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

# from sklearn.cluster import AgglomerativeClustering


basePath = "/data/Clustering/SleepDeprivation/RatN/Day2/"


# badChans = [14, 15, 16, 64]

# bestThetaChan = bestThetaChannel(
#     basePath, 1250, nChans=134, badChannels=badChans, saveThetaChan=1
# )


class SleepScore:
    lfpsampleRate = 1250
    nshanks = 8
    window = 4  # seconds
    slideby = 2  # seconds

    def __init__(self, basePath):
        self.sessionName = basePath.split("/")[-2] + basePath.split("/")[-1]
        self.basePath = basePath
        for file in os.listdir(basePath):
            if file.endswith(".eeg"):
                self.subname = file[:-4]
                self.filename = os.path.join(basePath, file)
                self.filePrefix = os.path.join(basePath, file[:-4])

    def thetaDeltaratio(self):
        nyq = 0.5 * self.lfpsampleRate
        highfreq = 300
        lowfreq = 275

        thetafile = basePath + self.sessionName + "_BestThetaChan.npy"
        bad_chans = np.load(self.filePrefix + "_badChans.npy", allow_pickle=True)
        basics = np.load(self.filePrefix + "_basics.npy", allow_pickle=True)
        chan_map = basics.item().get("channels")
        nChans = basics.item().get("nChans")

        thetaData = np.load(thetafile, allow_pickle=True)
        eegdata = np.memmap(self.filename, dtype="int16", mode="r")
        eegdata = np.memmap.reshape(eegdata, (int(len(eegdata) / nChans), nChans))

        chan_per_shank = nChans / self.nshanks
        # TODO top channel selection
        chan_multiple = int((nChans / self.nshanks) / self.nshanks)
        chan_top = chan_map[:: self.nshanks * chan_multiple]  # select top channels
        chan_bottom = chan_map[
            self.nshanks * chan_multiple - 1 :: self.nshanks * chan_multiple
        ]  # select bottom channels
        chan_map_select = np.union1d(chan_top, chan_bottom)

        # filtering for high frequency band
        chan_good = np.setdiff1d(chan_map_select, bad_chans)
        lfp_req = eegdata[:, chan_good]
        b, a = sg.butter(3, [lowfreq / nyq, highfreq / nyq], btype="bandpass")
        yf = sg.filtfilt(b, a, lfp_req, axis=0)

        # windowing signal
        frames = np.arange(
            0,
            len(eegdata) - self.window * self.lfpsampleRate,
            self.slideby * self.lfpsampleRate,
        )

        emg_lfp = []
        for i in range(len(frames)):

            start_frame = i * self.slideby * self.lfpsampleRate
            end_frame = start_frame + self.window * self.lfpsampleRate
            temp = (yf[start_frame:end_frame, :]).T
            corr_chan = np.corrcoef(temp)
            corr_all = corr_chan[np.tril_indices(len(corr_chan), k=-1)]
            emg_lfp.append(np.sum(corr_all))

        f, t, sxx = sg.spectrogram(
            thetaData,
            fs=self.lfpsampleRate,
            nperseg=self.window * self.lfpsampleRate,
            noverlap=self.slideby * self.lfpsampleRate,
        )

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
            (theta_delta_ratio, np.reshape(emg_lfp, (len(emg_lfp), 1)))
        )

        model = GaussianHMM(n_components=4, n_iter=100).fit(feature_comb)
        hidden_states = model.predict(feature_comb)
        # mus = np.squeeze(model.means_)
        # sigmas = np.squeeze(np.sqrt(model.covars_))
        # transmat = np.array(model.transmat_)

        # idx = np.argsort(mus)
        # mus = mus[idx]
        # sigmas = sigmas[idx]
        # transmat = transmat[idx, :][:, idx]

        # state_dict = {}
        # states = [i for i in range(4)]
        # for i in idx:
        #     state_dict[idx[i]] = states[i]

        # relabeled_states = [state_dict[h] for h in hidden_states]
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

        plt.clf()

        plt.imshow(
            sxx2,
            cmap="YlGn",
            aspect="auto",
            extent=[0, len(t), 0, 25.0],
            origin="lower",
            vmin=-500,
            vmax=140000,
            interpolation="mitchell",
        )
        # plt.pcolormesh(t / 3600, f, sxx, cmap="copper", vmax=30)

        # plt.plot(theta_delta_ratio)
        plt.plot((theta_delta_smooth + 5) * 2, "r", linewidth=2)
        plt.plot(relabeled_states + 4, color="#3fa8d5", linewidth=3)
        plt.plot(emg_lfp * 50)
        # plt.plot(hidden_states, "r")


basePath = [
    "/data/Clustering/SleepDeprivation/RatJ/Day1/",
    # "/data/Clustering/SleepDeprivation/RatJ/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day1/",
    # "/data/Clustering/SleepDeprivation/RatK/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day3/",
    # "/data/Clustering/SleepDeprivation/RatK/Day4/",
    # "/data/Clustering/SleepDeprivation/RatN/Day1/",
    # "/data/Clustering/SleepDeprivation/RatN/Day2/",
    # "/data/Clustering/SleepDeprivation/RatN/Day3/",
    # "/data/Clustering/SleepDeprivation/RatN/Day4/",
]
nSessions = len(basePath)
