#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 11:12:30 2019

@author: bapung
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sg
import scipy.stats as stats
import scipy.ndimage as gf
from OsCheck import DataDirPath, figDirPath, RawDataPath
import readOpenEphys as openEphys
# import scipy.signal as sg
# import scipy.stats as stats
# from scipy.signal import hilbert
#from SpectralAnalysis import lfpSpectMaze

#import seaborn as sns


sourceDir = RawDataPath() + 'SleepDeprivation/Beatrice/PRE/2019-04-22_04-10-57/'
filename = sourceDir + '100_ADC4.continuous'


data= openEphys.load(filename)

lfp = data['data']
eeg = lfp[0:-1:24]

#f,t,spect = sg.spectrogram(eeg, 1250,nperseg=2048*4, noverlap=2048*3)
#spect = gf.gaussian_filter(spect, sigma= 1)

plt.clf()
plt.plot(eeg)
#plt.pcolormesh(t, f[10:575], spect[10:575,:],cmap = 'hot', vmin=60, vmax=2050)
#with open(filename, 'rb') as f:
#        # Read header info, file length, and number of records
#        header = f.read(1024)

#file = loadContinuous(sourceDir)
#data = file.read(5)
