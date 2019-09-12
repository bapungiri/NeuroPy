#%%====== Import statements==========
import numpy as np
import matplotlib.pyplot as plt
from SpectralAnalysis import bestRippleChannel, bestThetaChannel, lfpSpectrogram
from lfpDetect import swr
import os

# %load_ext autoreload
# %autoreload 2


# %% ======== Theta detection testuing==============
basePath = (
    "/home/bapung/Documents/ClusteringHub/EEGAnlaysis/RatK/RatK_2019-08-06_03-44-01/"
)
nChans = 134
# subject = ''
# fileName = basePath + subject + '/' + subject + '.eeg'
subname = os.path.basename(os.path.normpath(basePath))
fileName = basePath + subname + ".eeg"
reqChan = 33
b1 = np.memmap(fileName, dtype="int16", mode="r")
ThetaExtract = b1[reqChan::nChans]

np.save(basePath + subname + "_BestThetaChan.npy", ThetaExtract)


subname = os.path.basename(os.path.normpath(basePath))
bestThetaCheck = basePath + subname + "_BestThetaChan.npy"
thetasec = np.load(bestThetaCheck)

T = 1 / 1250
N = len(thetasec)
fftTheta = np.fft.fft(thetasec)
xf = np.linspace(0.0, 1.0 / (2.0 * T), N / 2)

plt.clf()
plt.subplot(2, 1, 1)
plt.plot(xf, 2.0 / N * np.abs(fftTheta[: N // 2]))

plt.subplot(2, 1, 2)
plt.plot(thetasec[0 : 5 * 1250])


# fftTheta = np.fft.fft()

# sampleRate = 1250
# nChans = 134
# badChannels = np.arange(65, 134)

# bestTheta = bestThetaChannel(
#     basePath, sampleRate, nChans, badChannels, saveThetaChan=1)

# sxx, f, t, sample = lfpSpectrogram(
#     basePath, sampleRate, nChans, 64, loadfrom=0)
# sxx = sxx[0:500, :]
# sxx = np.flip(sxx)

# plt.clf()
# plt.imshow(
#     sxx, extent=[0, 10, f[0], f[500]], aspect="auto", vmax=70000, cmap="YlGn")
# plt.ylabel("Frequency (Hz)")
# plt.xlabel("Time (h)")

# bestChannel = bestRippleChannel(basePath, sampleRate, nChans, badChannels)
# ripples = swr(basePath, bestChannel[0], sampleRate, nChans)

# rpple_times = ripples[0]
# ripple_start = rpple_times[:, 0]

# ripple_counts, bin_edges = np.histogram(ripple_start, bins=14)

#%% ========== Ripple Detection ============

# basePath = (
#     "/home/bapung/Documents/ClusteringHub/EEGAnlaysis/RatK/RatK_2019-08-06_03-44-01/"
# )

basePath = (
    "/home/bapung/Documents/ClusteringHub/EEGAnlaysis/RatK/RatK_2019-08-08_04-00-00/"
)

nChans = 134
sRate = 1250
badChannels = np.arange(65, 134)
RippleTry = bestRippleChannel(
    basePath, sampleRate=sRate, nChans=nChans, badChannels=badChannels, saveRippleChan=1
)

# subname = os.path.basename(os.path.normpath(basePath))


# fileName = basePath + subname + "_BestRippleChans.npy"
# lfpCA1 = np.load(fileName)

ripples = swr(basePath, sRate=sRate, PlotRippleStat=1)
# ex = ripples[1]
# ex = ex["example_ripples"][0]

# flat_ripples = [item for sublist in ex for item in sublist]

# dt = 1 / 1250
# N = len(flat_ripples)
# Pxx = np.abs(np.fft.fft(flat_ripples))
# freq = np.fft.fftfreq(N, dt)

# plt.clf()
# plt.plot(flat_ripples)
# plt.plot(freq[: N // 2], (2 / N) * Pxx[: N // 2])


#%%
