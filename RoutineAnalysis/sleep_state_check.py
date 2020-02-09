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
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AgglomerativeClustering

# from sklearn.cluster import AgglomerativeClustering


basePath = "/data/Clustering/SleepDeprivation/RatN/Day2/"


# badChans = [14, 15, 16, 64]

# bestThetaChan = bestThetaChannel(
#     basePath, 1250, nChans=134, badChannels=badChans, saveThetaChan=1
# )


SampFreq = 1250
nyq = 0.5 * 1250


for file in os.listdir(basePath):
    if file.endswith(".eeg"):
        print(file)
        sessionName = file[:-4]
        eegfile = os.path.join(basePath, file)
        filePrefix = os.path.join(basePath, file[:-4])

filename = basePath + sessionName + "_BestThetaChan.npy"
data = np.load(filename, allow_pickle=True)
# signal = data
thetaData = data


sampleRate = 1250
N = len(thetaData)
eegdata = np.memmap(eegfile, dtype="int16", mode="r")
eegdata = np.memmap.reshape(eegdata, (int(len(eegdata) / 134), 134))
basics = np.load(filePrefix + "_basics.npy", allow_pickle=True)
chan_map = basics.item().get("channels")

chan_top = chan_map[::16]
chan_bottom = chan_map[15::16]
chan_map_select = np.union1d(chan_top, chan_bottom)
bad_chans = np.load(filePrefix + "_badChans.npy", allow_pickle=True)

chan_good = np.setdiff1d(chan_map_select, bad_chans)
lfp_req = eegdata[:, chan_good]
b, a = sg.butter(3, [250 / nyq, 300 / nyq], btype="bandpass")
yf = sg.filtfilt(b, a, lfp_req, axis=0)


frames = np.arange(0, len(eegdata) - 10 * sampleRate, 5 * sampleRate)

emg_lfp = []
for i in range(len(frames)):

    # duration = 10  # sd
    # lfpCA1 = np.memmap(
    #     eegfile,
    #     dtype="int16",
    #     mode="r",
    #     offset=2 * i * 5 * sampleRate * 134,
    #     shape=sampleRate * duration * 134,
    #
    # lfpCA1 = np.memmap.reshape(lfpCA1, (int(len(lfpCA1) / 134), 134))
    # lfpCA1 = lfpCA1[:, chan_good]

    # b, a = sg.butter(3, [250 / nyq, 300 / nyq], btype="bandpass")
    # yf = sg.filtfilt(b, a, lfpCA1, axis=0)

    start_frame = i * 5 * sampleRate
    end_frame = i * 5 * sampleRate + 10 * sampleRate
    temp = (yf[start_frame:end_frame, :]).T
    corr_chan = np.corrcoef(temp)
    corr_all = corr_chan[np.tril_indices(len(corr_chan), k=-1)]
    emg_lfp.append(np.mean(corr_all))


# N = 500
# t = np.linspace(0, 1, N)
# s = 0.7*np.sin(8 * 2 * np.pi * t)+0.5*np.sin(30 * 2 * np.pi * t) + 0.3 * np.sin(60 * 2 * np.pi * t)+0.2*np.random.randn(500)
# dt = t[1]-t[0]

# fftSig = np.abs(np.fft.fft(thetaData))
# freq = np.fft.fftfreq(N, 1/sampleRate)

f, t, sxx = sg.spectrogram(
    thetaData, fs=sampleRate, nperseg=10 * sampleRate, noverlap=5 * sampleRate
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
feature_comb = np.hstack((theta_delta_ratio, np.reshape(emg_lfp, (len(emg_lfp), 1))))

# cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
# cluster.fit_predict(feature_comb)
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
# # TODO coherence among multiple channels from different shanks
# theta_ratio_dist, bin_edges = np.histogram(theta_delta_ratio,bins=100)

# plt.plot(theta_gamma_ratio,theta_delta_ratio,'.')
# plt.plot(freq[:N//2], (2/N)*fftSig[:N//2])

relabeled_states = np.array(relabeled_states)
# sleep states labelling

# sleep_stages = []
# for i in range(4):

#     sleep_state = np.where(relabeled_states == i, 1, 0)
#     sleep_state = np.diff(sleep_state)
#     state_start = np.where(sleep_state == 1)[0]
#     state_end = np.where(sleep_state == -1)[0]
#     state_label = i * np.ones(len(state_start))
#     firstPass = np.vstack((t[state_start], t[state_end], state_label)).T
#     sleep_stages.extend(firstPass)

# sleep_stages = np.asarray(sleep_stages)
# sleep_stages = sleep_stages[sleep_stages[:, 0].argsort()]

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
plt.plot(stats.zscore(theta_sxx), stats.zscore(delta_sxx), ".")
# plt.plot(hidden_states, "r")
fig = plt.figure(1)
ax = fig.gca(projection="3d")

# z_line = np.linspace(0, 15, 1000)
# x_line = np.cos(z_line)
# y_line = np.sin(z_line)

N = len(theta_sxx)
a = np.reshape(stats.zscore(theta_sxx), (N, 1))
b = np.reshape(stats.zscore(theta_sxx), (N, 1))
c = np.reshape(emg_lfp, (N, 1))
feature_comb2 = np.hstack((a, b, c))
cluster = AgglomerativeClustering(n_clusters=3, affinity="euclidean", linkage="ward")
cluster.fit_predict(feature_comb2)

ax.scatter(a, b, c, c=cluster.labels_, cmap="rainbow")
plt.show()

