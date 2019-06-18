#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 13:10:11 2019

@author: bapung
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack as ft
import scipy.ndimage as smth
import scipy.signal as sg
import scipy.stats as stat
from SpectralAnalysis import bestThetaChannel
# import seaborn as sns

filename = '/data/Clustering/SleepDeprivation/RatJ/Day1/RatJ_2019-05-31_03-55-36/experiment1/recording1/continuous/Rhythm_FPGA-100.0/continuous.eeg'

SampFreq = 1250
#frames = RecInfo['behavFrames']
#behav = RecInfo['behav']
nChans = 75

badChannels = [1,3,6,7] + np.arange(65,76).tolist()
goodTheta= bestThetaChannel(filename,SampFreq,nChans,badChannels)

b1 = np.memmap(filename, dtype='int16', mode='r', shape=(SampFreq * 60 * 100,nChans))
goodThetalfp = b1[:,goodTheta]

plt.clf()

for chan in range(len(goodThetalfp[0,:])):


    eegnrem1= goodThetalfp[:,chan]
    # sos = sg.butter(3, 100, btype='low', fs=SampFreq, output='sos')
    # yf = sg.sosfilt(sos, eegnrem1)
    yf = ft.fft(eegnrem1) / len(eegnrem1)
    xf = np.linspace(0.0, SampFreq / 2, len(eegnrem1) // 2)
    y1 = 2.0 / (len(xf)) * np.abs(yf[:len(eegnrem1) // 2])
    y1 = smth.gaussian_filter(y1, 100)


    #plt.subplot()
    plt.plot(xf, y1,label = chan)
    plt.xlim([1, 30])
#    plt.ylim([0, 0.000005])
    plt.xlabel('Frequency')
    plt.ylabel('Power')

