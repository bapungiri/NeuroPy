import numpy as np
import matplotlib.pyplot as plt
from SpectralAnalysis import bestRippleChannel, bestThetaChannel, lfpSpectrogram
from lfpDetect import swr
import os


basePath = (
    "/home/bapung/Documents/ClusteringHub/EEGAnlaysis/RatK/RatK_2019-08-06_03-44-01/"
)
# subject = ''
# fileName = basePath + subject + '/' + subject + '.eeg'

subname = os.path.basename(os.path.normpath(basePath))
bestThetaCheck = basePath + subname + "_BestThetaChan.npy"
thetasec = np.load(bestThetaCheck)

fftTheta = np.fft.fft(thetasec)
xf = np.linspace(0.0, 1.0/(2.0*T), N/2)

fig, ax = plt.subplots()
ax.plot(xf, 2.0/N * np.abs(yf[:N//2]))

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
