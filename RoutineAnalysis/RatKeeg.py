import numpy as np
import matplotlib.pyplot as plt
from SpectralAnalysis import bestRippleChannel, bestThetaChannel, lfpSpectrogram
from lfpDetect import swr

basePath = '/home/bapung/Documents/ClusteringHub/EEGAnlaysis/RatK/'
subject = 'RatK_2019-08-06_03-44-01'
fileName = basePath + subject + '/' + subject + '.eeg'

sampleRate = 1250
nChans = 134
badChannels = np.arange(129, 135)

bestTheta = bestThetaChannel(fileName, sampleRate, nChans, badChannels)
sxx, f, t = lfpSpectrogram(fileName, sampleRate, nChans, bestTheta[0])
plt.imshow(sxx, aspect='auto')


bestChannel = bestRippleChannel(fileName, sampleRate, nChans, badChannels)
ripples = swr(fileName, bestChannel[0], sampleRate, nChans)

rpple_times = ripples[0]
ripple_start = rpple_times[:, 0]

ripple_counts, bin_edges = np.histogram(ripple_start, bins=14)
