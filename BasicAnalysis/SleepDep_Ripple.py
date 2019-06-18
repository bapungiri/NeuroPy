#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 18:53:31 2019

@author: bapung
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 16:32:41 2019

@author: bapung
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stat
import scipy.signal as sg
import scipy.fftpack as ft
import scipy.ndimage as smth
import seaborn as sns


#filename ='/data/Clustering/SleepDeprivation/RatJ/Day1/RatJ_2019-05-31_03-55-36/experiment1/recording1/continuous/Rhythm_FPGA-100.0/continuous.eeg'

filename = '/data/DataGen/SleepDeprivation/RatJDay1.npy'

SampFreq = 1250
#frames = RecInfo['behavFrames']
#behav = RecInfo['behav']
nChans = 75
ReqChan = 28

#offsetP = ((nREMPeriod - behav[2, 0]) // 1e6) * SampFreq + \
#    int(np.diff(frames[0, :])) + int(np.diff(frames[1, :]))


nyq = 0.5 * SampFreq
offsetp = (ReqChan-1)
signal = np.load(filename)
signal = signal[0:1250*3600*3]
signal = np.array(signal, dtype = np.float) # convert data to float


sos = sg.butter(3, [150/nyq,240/nyq], btype='bandpass', fs=SampFreq, output='sos')
yf = sg.sosfilt(sos,signal)

squared_signal = np.square(yf)
zscoreSignal = stat.zscore(squared_signal)

analytic_signal = sg.hilbert(yf)
amplitude_envelope = stat.zscore(np.abs(analytic_signal))

zscoreSignal = amplitude_envelope
ThreshSignal= np.diff(np.where(zscoreSignal>2,1,0))
start_ripple = np.argwhere(ThreshSignal ==1)
stop_ripple = np.argwhere(ThreshSignal ==-1)

ripple_duration = (stop_ripple-start_ripple)*(1000/1250)


print('Number of detected Events = ' ,len(start_ripple))

plt.clf()
plt.plot(signal[int(start_ripple[3]):int(stop_ripple[3])])
#plt.plot(zscoreSignal)
#plt.plot(amplitude_envelope, 'r')
#plt.title('Example Ripple')


#yf = ft.fft(yf) / len(eeg1)
#xf = np.linspace(0.0, SampFreq / 2, len(eegnrem1) // 2)
#y1 = 2.0 / (len(xf)) * np.abs(yf[:len(eegnrem1) // 2])
#y1 = smth.gaussian_filter(y1, 8)







#b1 = np.memmap(filename, dtype='int16', mode='r',
#               offset= 1 * (ReqChan - 1) * 2, shape=(1, nChans * SampFreq * 3600*15))
#eegnrem1 = b1[0, ::nChans]
#eegnrem1 = eegnrem1[0::24]
#sos = sg.butter(3, 100, btype='low', fs=SampFreq, output='sos')
#yf = sg.sosfilt(sos, eegnrem1)
#yf = ft.fft(yf) / len(eegnrem1)
#xf = np.linspace(0.0, SampFreq / 2, len(eegnrem1) // 2)
#y1 = 2.0 / (len(xf)) * np.abs(yf[:len(eegnrem1) // 2])
#y1 = smth.gaussian_filter(y1, 8)


#np.save('/data/DataGen/SleepDeprivation/RatJDay1',eegnrem1)


#f, t, Sxx = sg.spectrogram(eegnrem1, SampFreq, nperseg = 1250*6 ,noverlap =1250*5 , nfft= 8000)
#
#Sxx= stat.zscore(Sxx)
#Sxx= smth.gaussian_filter(Sxx, sigma=1.5)
#
#plt.clf()
#plt.ioff()
#plt.subplot(2,1,1)
##sns.heatmap(Sxx)
#plt.pcolormesh(t/3600, f, Sxx, cmap = 'copper', vmax = 30)
#plt.ylim([0, 40])
#plt.savefig('foo.png')
#plt.close(fig)
#filename ='/data/Clustering/SleepDeprivation/RatJ/Day2/RatJ_2019-06-02_03-59-19/experiment1/recording1/continuous/Rhythm_FPGA-100.0/continuous.eeg'
#
#SampFreq = 1250
##frames = RecInfo['behavFrames']
##behav = RecInfo['behav']
#nChans = 67
#ReqChan = 61
#
##offsetP = ((nREMPeriod - behav[2, 0]) // 1e6) * SampFreq + \
##    int(np.diff(frames[0, :])) + int(np.diff(frames[1, :]))
#
#offsetp = (ReqChan-1)
#
#b1 = np.memmap(filename, dtype='int16', mode='r',
#               offset= 1 * (ReqChan - 1) * 2, shape=(1, nChans * SampFreq * 3600*15))
#eegnrem1 = b1[0, ::nChans]
##eegnrem1 = eegnrem1[0::24]
##sos = sg.butter(3, 100, btype='low', fs=SampFreq, output='sos')
##yf = sg.sosfilt(sos, eegnrem1)
##yf = ft.fft(yf) / len(eegnrem1)
##xf = np.linspace(0.0, SampFreq / 2, len(eegnrem1) // 2)
##y1 = 2.0 / (len(xf)) * np.abs(yf[:len(eegnrem1) // 2])
##y1 = smth.gaussian_filter(y1, 8)
#
#f, t, Sxx = sg.spectrogram(eegnrem1, SampFreq, nperseg = 1250*10 ,noverlap =1250*2 )
#
#plt.subplot(2,1,2)
#plt.pcolormesh(t/3600, f, Sxx, cmap = 'copper', vmax = 200500)
#plt.ylim([0, 40])