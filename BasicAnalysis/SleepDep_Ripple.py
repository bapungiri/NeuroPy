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




nyq = 0.5 * SampFreq
offsetp = (ReqChan-1)
signal = np.load(filename)
signal = signal[0:1250*14*3600]
signal = np.array(signal, dtype = np.float) # convert data to float

#zscoreSignal = stat.zscore(signal)

b,a= sg.butter(3, [150/nyq,240/nyq],btype='bandpass')
yf = sg.filtfilt(b,a,signal)

squared_signal = np.square(yf)
normsquaredsignal = stat.zscore(squared_signal)

#getting an envelope of the signal
analytic_signal = sg.hilbert(yf)
amplitude_envelope = stat.zscore(np.abs(analytic_signal))
#

windowLength = SampFreq/SampFreq*11
window = np.ones((int(windowLength),))/windowLength


smoothSignal = sg.filtfilt(window,1,squared_signal, axis=0)
zscoreSignal = stat.zscore(smoothSignal)
#
ThreshSignal= np.diff(np.where(zscoreSignal>2,1,0))
start_ripple = np.argwhere(ThreshSignal ==1)
stop_ripple = np.argwhere(ThreshSignal ==-1)
##


firstPass= np.concatenate((start_ripple,stop_ripple),axis=1)


minInterRippleSamples = 30/1000*1250;
secondPass = []
ripple = firstPass[0]
for i in range(1,len(firstPass)):
	if firstPass[i,0] - ripple[1] < minInterRippleSamples:
		# Merging ripples
		ripple = [ripple[0],firstPass[i,1]]
	else:
		secondPass.append(ripple)
		ripple = firstPass[i]

secondPass.append(ripple)
secondPass= np.asarray(secondPass)

#delete ripples with less than threshold power
thirdPass = []
peakNormalizedPower = []
highThresholdFactor=5
for i in range(0,len(secondPass)):
	maxValue = max(zscoreSignal[secondPass[i,0]:secondPass[i,1]])
	if maxValue > highThresholdFactor:
		thirdPass.append(secondPass[i])
		peakNormalizedPower.append(maxValue)

thirdPass= np.asarray(thirdPass)

ripple_duration = np.diff(thirdPass,axis=1)/1250*1000


#delete very short ripples
shortRipples = np.where(ripple_duration < 20)[0]
thirdPass =np.delete(thirdPass,shortRipples,0)



ripple_rate,_ = np.histogram(thirdPass[:,0],140)







#
#
#print('Number of detected Events = ' ,len(start_ripple))

plt.clf()

#plt.plot(signal[int(thirdPass[10,0])-125:int(thirdPass[10,1])+125])
plt.plot(ripple_rate)
#plt.plot(zscoreSignal,'r')
#plt.plot(amplitude_envelope,'k')
#plt.plot(amplitude_envelope,'r')
#plt.plot(amplitude_envelope, 'r')
#plt.title('Example Ripple')


#yf = ft.fft(yf) / len(eeg1)
#xf = np.linspace(0.0, SampFreq / 2, len(eegnrem1) // 2)
#y1 = 2.0 / (len(xf)) * np.abs(yf[:len(eegnrem1) // 2])
#y1 = smth.gaussian_filter(y1, 8)







