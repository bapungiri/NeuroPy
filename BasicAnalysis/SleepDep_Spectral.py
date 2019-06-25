#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 16:32:41 2019

@author: bapung
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack as ft
import scipy.ndimage as smth
import scipy.signal as sg
import scipy.stats as stat
import seaborn as sns

filename = '/data/Clustering/SleepDeprivation/RatJ/Day1/RatJ_2019-05-31_03-55-36/experiment1/recording1/continuous/Rhythm_FPGA-100.0/continuous.eeg'

SampFreq = 1250
#frames = RecInfo['behavFrames']
#behav = RecInfo['behav']
nChans = 75
ReqChan = 28

# offsetP = ((nREMPeriod - behav[2, 0]) // 1e6) * SampFreq + \
#    int(np.diff(frames[0, :])) + int(np.diff(frames[1, :]))

offsetp = (ReqChan-1)

b1 = np.memmap(filename, dtype='int16', mode='r',
               offset=1 * (ReqChan - 1) * 2, shape=(1, nChans * SampFreq * 3600 * 15))
eegnrem1 = b1[0, ::nChans]
# eegnrem1 = eegnrem1[0::24]
# sos = sg.butter(3, 100, btype='low', fs=SampFreq, output='sos')
# yf = sg.sosfilt(sos, eegnrem1)
# yf = ft.fft(yf) / len(eegnrem1)
# xf = np.linspace(0.0, SampFreq / 2, len(eegnrem1) // 2)
# y1 = 2.0 / (len(xf)) * np.abs(yf[:len(eegnrem1) // 2])
# y1 = smth.gaussi#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 16:32:41 2019

@author: bapung
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack as ft
import scipy.ndimage as smth
import scipy.signal as sg
import scipy.stats as stat
import seaborn as sns

filename = '/data/Clustering/SleepDeprivation/RatJ/Day1/RatJ_2019-05-31_03-55-36/experiment1/recording1/continuous/Rhythm_FPGA-100.0/continuous.eeg'

SampFreq = 1250
#frames = RecInfo['behavFrames']
#behav = RecInfo['behav']
nChans = 75
ReqChan = 28

# offsetP = ((nREMPeriod - behav[2, 0]) // 1e6) * SampFreq + \
#    int(np.diff(frames[0, :])) + int(np.diff(frames[1, :]))

offsetp = (ReqChan-1)

b1 = np.memmap(filename, dtype='int16', mode='r',
               offset=1 * (ReqChan - 1) * 2, shape=(1, nChans * SampFreq * 3600 * 15))
eegnrem1 = b1[0, ::nChans]
# eegnrem1 = eegnrem1[0::24]
# sos = sg.butter(3, 100, btype='low', fs=SampFreq, output='sos')
# yf = sg.sosfilt(sos, eegnrem1)
# yf = ft.fft(yf) / len(eegnrem1)
# xf = np.linspace(0.0, SampFreq / 2, len(eegnrem1) // 2)
# y1 = 2.0 / (len(xf)) * np.abs(yf[:len(eegnrem1) // 2])
# y1 = smth.gaussian_filter(y1, 8)

f, t, Sxx = sg.spectrogram(
    eegnrem1, SampFreq, nperseg=1250 * 6, noverlap=1250 * 5, nfft=8000)

Sxx = stat.zscore(Sxx)
Sxx = smth.gaussian_filter(Sxx, sigma=1.5)

fig = plt.figure()
plt.subplot(2, 1, 1)
# sns.heatmap(Sxx)
plt.pcolormesh(t/3600, f, Sxx, cmap='copper', vmax=30)
plt.ylim([0, 40])
plt.title('Day 1')
# plt.savefig('foo.png')
# plt.close(fig)
filename = '/data/Clustering/SleepDeprivation/RatJ/Day2/RatJ_2019-06-02_03-59-19/experiment1/recording1/continuous/Rhythm_FPGA-100.0/continuous.eeg'

SampFreq = 1250
# frames = RecInfo['behavFrames']
# behav = RecInfo['behav']
nChans = 67
ReqChan = 61

# offsetP = ((nREMPeriod - behav[2, 0]) // 1e6) * SampFreq + \
#    int(np.diff(frames[0, :])) + int(np.diff(frames[1, :]))

offsetp = (ReqChan - 1)

b1 = np.memmap(filename, dtype='int16', mode='r',
               offset=1 * (ReqChan - 1) * 2, shape=(1, nChans * SampFreq * 3600 * 15))
eegnrem1 = b1[0, ::nChans]
#eegnrem1 = eegnrem1[0::24]
#sos = sg.butter(3, 100, btype='low', fs=SampFreq, output='sos')
#yf = sg.sosfilt(sos, eegnrem1)
#yf = ft.fft(yf) / len(eegnrem1)
#xf = np.linspace(0.0, SampFreq / 2, len(eegnrem1) // 2)
#y1 = 2.0 / (len(xf)) * np.abs(yf[:len(eegnrem1) // 2])
#y1 = smth.gaussian_filter(y1, 8)

f, t, Sxx = sg.spectrogram(
    eegnrem1, SampFreq, nperseg=1250 * 6, noverlap=1250 * 5, nfft=8000)
Sxx = stat.zscore(Sxx)
Sxx = smth.gaussian_filter(Sxx, sigma=1.5)

plt.subplot(2, 1, 2)
plt.pcolormesh(t / 3600, f, Sxx, cmap='copper', vmax=30)
plt.ylim([0, 40])
plt.xlabel('Time (h)')
plt.ylabel('Frequency (Hz)')
plt.title('Day 2')


f, t, Sxx = sg.spe#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 16:32:41 2019

@author: bapung
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack as ft
import scipy.ndimage as smth
import scipy.signal as sg
import scipy.stats as stat
import seaborn as sns

filename = '/data/Clustering/SleepDeprivation/RatJ/Day1/RatJ_2019-05-31_03-55-36/experiment1/recording1/continuous/Rhythm_FPGA-100.0/continuous.eeg'

SampFreq = 1250
#frames = RecInfo['behavFrames']
#behav = RecInfo['behav']
nChans = 75
ReqChan = 28

# offsetP = ((nREMPeriod - behav[2, 0]) // 1e6) * SampFreq + \
#    int(np.diff(frames[0, :])) + int(np.diff(frames[1, :]))

offsetp = (ReqChan-1)

b1 = np.memmap(filename, dtype='int16', mode='r',
               offset=1 * (ReqChan - 1) * 2, shape=(1, nChans * SampFreq * 3600 * 15))
eegnrem1 = b1[0, ::nChans]
# eegnrem1 = eegnrem1[0::24]
# sos = sg.butter(3, 100, btype='low', fs=SampFreq, output='sos')
# yf = sg.sosfilt(sos, eegnrem1)
# yf = ft.fft(yf) / len(eegnrem1)
# xf = np.linspace(0.0, SampFreq / 2, len(eegnrem1) // 2)
# y1 = 2.0 / (len(xf)) * np.abs(yf[:len(eegnrem1) // 2])
# y1 = smth.gaussian_filter(y1, 8)

f, t, Sxx = sg.spectrogram(
    eegnrem1, SampFreq, nperseg=1250 * 6, noverlap=1250 * 5, nfft=8000)

Sxx = stat.zscore(Sxx)
Sxx = smth.gaussian_filter(Sxx, sigma=1.5)

fig = plt.figure()
plt.subplot(2, 1, 1)
# sns.heatmap(Sxx)
plt.pcolormesh(t/3600, f, Sxx, cmap='copper', vmax=30)
plt.ylim([0, 40])
plt.title('Day 1')
# plt.savefig('foo.png')
# plt.close(fig)
filename = '/data/Clustering/SleepDeprivation/RatJ/Day2/RatJ_2019-06-02_03-59-19/experiment1/recording1/continuous/Rhythm_FPGA-100.0/continuous.eeg'

SampFreq = 1250
# frames = RecInfo['behavFrames']
# behav = RecInfo['behav']
nChans = 67
ReqChan = 61

# offsetP = ((nREMPeriod - behav[2, 0]) // 1e6) * SampFreq + \
#    int(np.diff(frames[0, :])) + int(np.diff(frames[1, :]))

offsetp = (ReqChan - 1)

b1 = np.memmap(filename, dtype='int16', mode='r',
               offset=1 * (ReqChan - 1) * 2, shape=(1, nChans * SampFreq * 3600 * 15))
eegnrem1 = b1[0, ::nChans]
#eegnrem1 = eegnrem1[0::24]
#sos = sg.butter(3, 100, btype='low', fs=SampFreq, output='sos')
#yf = sg.sosfilt(sos, eegnrem1)
#yf = ft.fft(yf) / len(eegnrem1)
#xf = np.linspace(0.0, SampFreq / 2, len(eegnrem1) // 2)
#y1 = 2.0 / (len(xf)) * np.abs(yf[:len(eegnrem1) // 2])
#y1 = smth.gaussian_filter(y1, 8)

f, t, Sxx = sg.spectrogram(
    eegnrem1, SampFreq, nperseg=1250 * 6, noverlap=1250 * 5, nfft=8000)
Sxx = stat.zscore(Sxx)
Sxx = smth.gaussian_filter(Sxx, sigma=1.5)

plt.subplot(2, 1, 2)
plt.pcolormesh(t / 3600, f, Sxx, cmap='copper', vmax=30)
plt.ylim([0, 40])
plt.xlabel('Time (h)')
plt.ylabel('Frequency (Hz)')
plt.title('Day 2')

    eegnrem1, Samp#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 16:32:41 2019

@author: bapung
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack as ft
import scipy.ndimage as smth
import scipy.signal as sg
import scipy.stats as stat
import seaborn as sns

filename = '/data/Clustering/SleepDeprivation/RatJ/Day1/RatJ_2019-05-31_03-55-36/experiment1/recording1/continuous/Rhythm_FPGA-100.0/continuous.eeg'

SampFreq = 1250
#frames = RecInfo['behavFrames']
#behav = RecInfo['behav']
nChans = 75
ReqChan = 28

# offsetP = ((nREMPeriod - behav[2, 0]) // 1e6) * SampFreq + \
#    int(np.diff(frames[0, :])) + int(np.diff(frames[1, :]))

offsetp = (ReqChan-1)

b1 = np.memmap(filename, dtype='int16', mode='r',
               offset=1 * (ReqChan - 1) * 2, shape=(1, nChans * SampFreq * 3600 * 15))
eegnrem1 = b1[0, ::nChans]
# eegnrem1 = eegnrem1[0::24]
# sos = sg.butter(3, 100, btype='low', fs=SampFreq, output='sos')
# yf = sg.sosfilt(sos, eegnrem1)
# yf = ft.fft(yf) / len(eegnrem1)
# xf = np.linspace(0.0, SampFreq / 2, len(eegnrem1) // 2)
# y1 = 2.0 / (len(xf)) * np.abs(yf[:len(eegnrem1) // 2])
# y1 = smth.gaussian_filter(y1, 8)

f, t, Sxx = sg.spectrogram(
    eegnrem1, SampFreq, nperseg=1250 * 6, noverlap=1250 * 5, nfft=8000)

Sxx = stat.zscore(Sxx)
Sxx = smth.gaussian_filter(Sxx, sigma=1.5)

fig = plt.figure()
plt.subplot(2, 1, 1)
# sns.heatmap(Sxx)
plt.pcolormesh(t/3600, f, Sxx, cmap='copper', vmax=30)
plt.ylim([0, 40])
plt.title('Day 1')
# plt.savefig('foo.png')
# plt.close(fig)
filename = '/data/Clustering/SleepDeprivation/RatJ/Day2/RatJ_2019-06-02_03-59-19/experiment1/recording1/continuous/Rhythm_FPGA-100.0/continuous.eeg'

SampFreq = 1250
# frames = RecInfo['behavFrames']
# behav = RecInfo['behav']
nChans = 67
ReqChan = 61

# offsetP = ((nREMPeriod - behav[2, 0]) // 1e6) * SampFreq + \
#    int(np.diff(frames[0, :])) + int(np.diff(frames[1, :]))

offsetp = (ReqChan - 1)

b1 = np.memmap(filename, dtype='int16', mode='r',
               offset=1 * (ReqChan - 1) * 2, shape=(1, nChans * SampFreq * 3600 * 15))
eegnrem1 = b1[0, ::nChans]
#eegnrem1 = eegnrem1[0::24]
#sos = sg.butter(3, 100, btype='low', fs=SampFreq, output='sos')
#yf = sg.sosfilt(sos, eegnrem1)
#yf = ft.fft(yf) / len(eegnrem1)
#xf = np.linspace(0.0, SampFreq / 2, len(eegnrem1) // 2)
#y1 = 2.0 / (len(xf)) * np.abs(yf[:len(eegnrem1) // 2])
#y1 = smth.gaussian_filter(y1, 8)

f, t, Sxx = sg.spectrogram(
    eegnrem1, SampFreq, nperseg=1250 * 6, noverlap=1250 * 5, nfft=8000)
Sxx = stat.zscore(Sxx)
Sxx = smth.gaussian_filter(Sxx, sigma=1.5)

plt.subplot(2, 1, 2)
plt.pcolormesh(t / 3600, f, Sxx, cmap='copper', vmax=30)
plt.ylim([0, 40])
plt.xlabel('Time (h)')
plt.ylabel('Frequency (Hz)')
plt.title('Day 2')


Sxx = stat.zscore(#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 16:32:41 2019

@author: bapung
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack as ft
import scipy.ndimage as smth
import scipy.signal as sg
import scipy.stats as stat
import seaborn as sns

filename = '/data/Clustering/SleepDeprivation/RatJ/Day1/RatJ_2019-05-31_03-55-36/experiment1/recording1/continuous/Rhythm_FPGA-100.0/continuous.eeg'

SampFreq = 1250
#frames = RecInfo['behavFrames']
#behav = RecInfo['behav']
nChans = 75
ReqChan = 28

# offsetP = ((nREMPeriod - behav[2, 0]) // 1e6) * SampFreq + \
#    int(np.diff(frames[0, :])) + int(np.diff(frames[1, :]))

offsetp = (ReqChan-1)

b1 = np.memmap(filename, dtype='int16', mode='r',
               offset=1 * (ReqChan - 1) * 2, shape=(1, nChans * SampFreq * 3600 * 15))
eegnrem1 = b1[0, ::nChans]
# eegnrem1 = eegnrem1[0::24]
# sos = sg.butter(3, 100, btype='low', fs=SampFreq, output='sos')
# yf = sg.sosfilt(sos, eegnrem1)
# yf = ft.fft(yf) / len(eegnrem1)
# xf = np.linspace(0.0, SampFreq / 2, len(eegnrem1) // 2)
# y1 = 2.0 / (len(xf)) * np.abs(yf[:len(eegnrem1) // 2])
# y1 = smth.gaussian_filter(y1, 8)

f, t, Sxx = sg.spectrogram(
    eegnrem1, SampFreq, nperseg=1250 * 6, noverlap=1250 * 5, nfft=8000)

Sxx = stat.zscore(Sxx)
Sxx = smth.gaussian_filter(Sxx, sigma=1.5)

fig = plt.figure()
plt.subplot(2, 1, 1)
# sns.heatmap(Sxx)
plt.pcolormesh(t/3600, f, Sxx, cmap='copper', vmax=30)
plt.ylim([0, 40])
plt.title('Day 1')
# plt.savefig('foo.png')
# plt.close(fig)
filename = '/data/Clustering/SleepDeprivation/RatJ/Day2/RatJ_2019-06-02_03-59-19/experiment1/recording1/continuous/Rhythm_FPGA-100.0/continuous.eeg'

SampFreq = 1250
# frames = RecInfo['behavFrames']
# behav = RecInfo['behav']
nChans = 67
ReqChan = 61

# offsetP = ((nREMPeriod - behav[2, 0]) // 1e6) * SampFreq + \
#    int(np.diff(frames[0, :])) + int(np.diff(frames[1, :]))

offsetp = (ReqChan - 1)

b1 = np.memmap(filename, dtype='int16', mode='r',
               offset=1 * (ReqChan - 1) * 2, shape=(1, nChans * SampFreq * 3600 * 15))
eegnrem1 = b1[0, ::nChans]
#eegnrem1 = eegnrem1[0::24]
#sos = sg.butter(3, 100, btype='low', fs=SampFreq, output='sos')
#yf = sg.sosfilt(sos, eegnrem1)
#yf = ft.fft(yf) / len(eegnrem1)
#xf = np.linspace(0.0, SampFreq / 2, len(eegnrem1) // 2)
#y1 = 2.0 / (len(xf)) * np.abs(yf[:len(eegnrem1) // 2])
#y1 = smth.gaussian_filter(y1, 8)

f, t, Sxx = sg.spectrogram(
    eegnrem1, SampFreq, nperseg=1250 * 6, noverlap=1250 * 5, nfft=8000)
Sxx = stat.zscore(Sxx)
Sxx = smth.gaussian_filter(Sxx, sigma=1.5)

plt.subplot(2, 1, 2)
plt.pcolormesh(t / 3600, f, Sxx, cmap='copper', vmax=30)
plt.ylim([0, 40])
plt.xlabel('Time (h)')
plt.ylabel('Frequency (Hz)')
plt.title('Day 2')

Sxx = smth.gaussia#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 16:32:41 2019

@author: bapung
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack as ft
import scipy.ndimage as smth
import scipy.signal as sg
import scipy.stats as stat
import seaborn as sns

filename = '/data/Clustering/SleepDeprivation/RatJ/Day1/RatJ_2019-05-31_03-55-36/experiment1/recording1/continuous/Rhythm_FPGA-100.0/continuous.eeg'

SampFreq = 1250
#frames = RecInfo['behavFrames']
#behav = RecInfo['behav']
nChans = 75
ReqChan = 28

# offsetP = ((nREMPeriod - behav[2, 0]) // 1e6) * SampFreq + \
#    int(np.diff(frames[0, :])) + int(np.diff(frames[1, :]))

offsetp = (ReqChan-1)

b1 = np.memmap(filename, dtype='int16', mode='r',
               offset=1 * (ReqChan - 1) * 2, shape=(1, nChans * SampFreq * 3600 * 15))
eegnrem1 = b1[0, ::nChans]
# eegnrem1 = eegnrem1[0::24]
# sos = sg.butter(3, 100, btype='low', fs=SampFreq, output='sos')
# yf = sg.sosfilt(sos, eegnrem1)
# yf = ft.fft(yf) / len(eegnrem1)
# xf = np.linspace(0.0, SampFreq / 2, len(eegnrem1) // 2)
# y1 = 2.0 / (len(xf)) * np.abs(yf[:len(eegnrem1) // 2])
# y1 = smth.gaussian_filter(y1, 8)

f, t, Sxx = sg.spectrogram(
    eegnrem1, SampFreq, nperseg=1250 * 6, noverlap=1250 * 5, nfft=8000)

Sxx = stat.zscore(Sxx)
Sxx = smth.gaussian_filter(Sxx, sigma=1.5)

fig = plt.figure()
plt.subplot(2, 1, 1)
# sns.heatmap(Sxx)
plt.pcolormesh(t/3600, f, Sxx, cmap='copper', vmax=30)
plt.ylim([0, 40])
plt.title('Day 1')
# plt.savefig('foo.png')
# plt.close(fig)
filename = '/data/Clustering/SleepDeprivation/RatJ/Day2/RatJ_2019-06-02_03-59-19/experiment1/recording1/continuous/Rhythm_FPGA-100.0/continuous.eeg'

SampFreq = 1250
# frames = RecInfo['behavFrames']
# behav = RecInfo['behav']
nChans = 67
ReqChan = 61

# offsetP = ((nREMPeriod - behav[2, 0]) // 1e6) * SampFreq + \
#    int(np.diff(frames[0, :])) + int(np.diff(frames[1, :]))

offsetp = (ReqChan - 1)

b1 = np.memmap(filename, dtype='int16', mode='r',
               offset=1 * (ReqChan - 1) * 2, shape=(1, nChans * SampFreq * 3600 * 15))
eegnrem1 = b1[0, ::nChans]
#eegnrem1 = eegnrem1[0::24]
#sos = sg.butter(3, 100, btype='low', fs=SampFreq, output='sos')
#yf = sg.sosfilt(sos, eegnrem1)
#yf = ft.fft(yf) / len(eegnrem1)
#xf = np.linspace(0.0, SampFreq / 2, len(eegnrem1) // 2)
#y1 = 2.0 / (len(xf)) * np.abs(yf[:len(eegnrem1) // 2])
#y1 = smth.gaussian_filter(y1, 8)

f, t, Sxx = sg.spectrogram(
    eegnrem1, SampFreq, nperseg=1250 * 6, noverlap=1250 * 5, nfft=8000)
Sxx = stat.zscore(Sxx)
Sxx = smth.gaussian_filter(Sxx, sigma=1.5)

plt.subplot(2, 1, 2)
plt.pcolormesh(t / 3600, f, Sxx, cmap='copper', vmax=30)
plt.ylim([0, 40])
plt.xlabel('Time (h)')
plt.ylabel('Frequency (Hz)')
plt.title('Day 2')


fig = plt.figure()#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 16:32:41 2019

@author: bapung
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack as ft
import scipy.ndimage as smth
import scipy.signal as sg
import scipy.stats as stat
import seaborn as sns

filename = '/data/Clustering/SleepDeprivation/RatJ/Day1/RatJ_2019-05-31_03-55-36/experiment1/recording1/continuous/Rhythm_FPGA-100.0/continuous.eeg'

SampFreq = 1250
#frames = RecInfo['behavFrames']
#behav = RecInfo['behav']
nChans = 75
ReqChan = 28

# offsetP = ((nREMPeriod - behav[2, 0]) // 1e6) * SampFreq + \
#    int(np.diff(frames[0, :])) + int(np.diff(frames[1, :]))

offsetp = (ReqChan-1)

b1 = np.memmap(filename, dtype='int16', mode='r',
               offset=1 * (ReqChan - 1) * 2, shape=(1, nChans * SampFreq * 3600 * 15))
eegnrem1 = b1[0, ::nChans]
# eegnrem1 = eegnrem1[0::24]
# sos = sg.butter(3, 100, btype='low', fs=SampFreq, output='sos')
# yf = sg.sosfilt(sos, eegnrem1)
# yf = ft.fft(yf) / len(eegnrem1)
# xf = np.linspace(0.0, SampFreq / 2, len(eegnrem1) // 2)
# y1 = 2.0 / (len(xf)) * np.abs(yf[:len(eegnrem1) // 2])
# y1 = smth.gaussian_filter(y1, 8)

f, t, Sxx = sg.spectrogram(
    eegnrem1, SampFreq, nperseg=1250 * 6, noverlap=1250 * 5, nfft=8000)

Sxx = stat.zscore(Sxx)
Sxx = smth.gaussian_filter(Sxx, sigma=1.5)

fig = plt.figure()
plt.subplot(2, 1, 1)
# sns.heatmap(Sxx)
plt.pcolormesh(t/3600, f, Sxx, cmap='copper', vmax=30)
plt.ylim([0, 40])
plt.title('Day 1')
# plt.savefig('foo.png')
# plt.close(fig)
filename = '/data/Clustering/SleepDeprivation/RatJ/Day2/RatJ_2019-06-02_03-59-19/experiment1/recording1/continuous/Rhythm_FPGA-100.0/continuous.eeg'

SampFreq = 1250
# frames = RecInfo['behavFrames']
# behav = RecInfo['behav']
nChans = 67
ReqChan = 61

# offsetP = ((nREMPeriod - behav[2, 0]) // 1e6) * SampFreq + \
#    int(np.diff(frames[0, :])) + int(np.diff(frames[1, :]))

offsetp = (ReqChan - 1)

b1 = np.memmap(filename, dtype='int16', mode='r',
               offset=1 * (ReqChan - 1) * 2, shape=(1, nChans * SampFreq * 3600 * 15))
eegnrem1 = b1[0, ::nChans]
#eegnrem1 = eegnrem1[0::24]
#sos = sg.butter(3, 100, btype='low', fs=SampFreq, output='sos')
#yf = sg.sosfilt(sos, eegnrem1)
#yf = ft.fft(yf) / len(eegnrem1)
#xf = np.linspace(0.0, SampFreq / 2, len(eegnrem1) // 2)
#y1 = 2.0 / (len(xf)) * np.abs(yf[:len(eegnrem1) // 2])
#y1 = smth.gaussian_filter(y1, 8)

f, t, Sxx = sg.spectrogram(
    eegnrem1, SampFreq, nperseg=1250 * 6, noverlap=1250 * 5, nfft=8000)
Sxx = stat.zscore(Sxx)
Sxx = smth.gaussian_filter(Sxx, sigma=1.5)

plt.subplot(2, 1, 2)
plt.pcolormesh(t / 3600, f, Sxx, cmap='copper', vmax=30)
plt.ylim([0, 40])
plt.xlabel('Time (h)')
plt.ylabel('Frequency (Hz)')
plt.title('Day 2')

plt.subplot(2, 1, #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 16:32:41 2019

@author: bapung
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack as ft
import scipy.ndimage as smth
import scipy.signal as sg
import scipy.stats as stat
import seaborn as sns

filename = '/data/Clustering/SleepDeprivation/RatJ/Day1/RatJ_2019-05-31_03-55-36/experiment1/recording1/continuous/Rhythm_FPGA-100.0/continuous.eeg'

SampFreq = 1250
#frames = RecInfo['behavFrames']
#behav = RecInfo['behav']
nChans = 75
ReqChan = 28

# offsetP = ((nREMPeriod - behav[2, 0]) // 1e6) * SampFreq + \
#    int(np.diff(frames[0, :])) + int(np.diff(frames[1, :]))

offsetp = (ReqChan-1)

b1 = np.memmap(filename, dtype='int16', mode='r',
               offset=1 * (ReqChan - 1) * 2, shape=(1, nChans * SampFreq * 3600 * 15))
eegnrem1 = b1[0, ::nChans]
# eegnrem1 = eegnrem1[0::24]
# sos = sg.butter(3, 100, btype='low', fs=SampFreq, output='sos')
# yf = sg.sosfilt(sos, eegnrem1)
# yf = ft.fft(yf) / len(eegnrem1)
# xf = np.linspace(0.0, SampFreq / 2, len(eegnrem1) // 2)
# y1 = 2.0 / (len(xf)) * np.abs(yf[:len(eegnrem1) // 2])
# y1 = smth.gaussian_filter(y1, 8)

f, t, Sxx = sg.spectrogram(
    eegnrem1, SampFreq, nperseg=1250 * 6, noverlap=1250 * 5, nfft=8000)

Sxx = stat.zscore(Sxx)
Sxx = smth.gaussian_filter(Sxx, sigma=1.5)

fig = plt.figure()
plt.subplot(2, 1, 1)
# sns.heatmap(Sxx)
plt.pcolormesh(t/3600, f, Sxx, cmap='copper', vmax=30)
plt.ylim([0, 40])
plt.title('Day 1')
# plt.savefig('foo.png')
# plt.close(fig)
filename = '/data/Clustering/SleepDeprivation/RatJ/Day2/RatJ_2019-06-02_03-59-19/experiment1/recording1/continuous/Rhythm_FPGA-100.0/continuous.eeg'

SampFreq = 1250
# frames = RecInfo['behavFrames']
# behav = RecInfo['behav']
nChans = 67
ReqChan = 61

# offsetP = ((nREMPeriod - behav[2, 0]) // 1e6) * SampFreq + \
#    int(np.diff(frames[0, :])) + int(np.diff(frames[1, :]))

offsetp = (ReqChan - 1)

b1 = np.memmap(filename, dtype='int16', mode='r',
               offset=1 * (ReqChan - 1) * 2, shape=(1, nChans * SampFreq * 3600 * 15))
eegnrem1 = b1[0, ::nChans]
#eegnrem1 = eegnrem1[0::24]
#sos = sg.butter(3, 100, btype='low', fs=SampFreq, output='sos')
#yf = sg.sosfilt(sos, eegnrem1)
#yf = ft.fft(yf) / len(eegnrem1)
#xf = np.linspace(0.0, SampFreq / 2, len(eegnrem1) // 2)
#y1 = 2.0 / (len(xf)) * np.abs(yf[:len(eegnrem1) // 2])
#y1 = smth.gaussian_filter(y1, 8)

f, t, Sxx = sg.spectrogram(
    eegnrem1, SampFreq, nperseg=1250 * 6, noverlap=1250 * 5, nfft=8000)
Sxx = stat.zscore(Sxx)
Sxx = smth.gaussian_filter(Sxx, sigma=1.5)

plt.subplot(2, 1, 2)
plt.pcolormesh(t / 3600, f, Sxx, cmap='copper', vmax=30)
plt.ylim([0, 40])
plt.xlabel('Time (h)')
plt.ylabel('Frequency (Hz)')
plt.title('Day 2')

# sns.heatmap(Sxx)#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 16:32:41 2019

@author: bapung
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack as ft
import scipy.ndimage as smth
import scipy.signal as sg
import scipy.stats as stat
import seaborn as sns

filename = '/data/Clustering/SleepDeprivation/RatJ/Day1/RatJ_2019-05-31_03-55-36/experiment1/recording1/continuous/Rhythm_FPGA-100.0/continuous.eeg'

SampFreq = 1250
#frames = RecInfo['behavFrames']
#behav = RecInfo['behav']
nChans = 75
ReqChan = 28

# offsetP = ((nREMPeriod - behav[2, 0]) // 1e6) * SampFreq + \
#    int(np.diff(frames[0, :])) + int(np.diff(frames[1, :]))

offsetp = (ReqChan-1)

b1 = np.memmap(filename, dtype='int16', mode='r',
               offset=1 * (ReqChan - 1) * 2, shape=(1, nChans * SampFreq * 3600 * 15))
eegnrem1 = b1[0, ::nChans]
# eegnrem1 = eegnrem1[0::24]
# sos = sg.butter(3, 100, btype='low', fs=SampFreq, output='sos')
# yf = sg.sosfilt(sos, eegnrem1)
# yf = ft.fft(yf) / len(eegnrem1)
# xf = np.linspace(0.0, SampFreq / 2, len(eegnrem1) // 2)
# y1 = 2.0 / (len(xf)) * np.abs(yf[:len(eegnrem1) // 2])
# y1 = smth.gaussian_filter(y1, 8)

f, t, Sxx = sg.spectrogram(
    eegnrem1, SampFreq, nperseg=1250 * 6, noverlap=1250 * 5, nfft=8000)

Sxx = stat.zscore(Sxx)
Sxx = smth.gaussian_filter(Sxx, sigma=1.5)

fig = plt.figure()
plt.subplot(2, 1, 1)
# sns.heatmap(Sxx)
plt.pcolormesh(t/3600, f, Sxx, cmap='copper', vmax=30)
plt.ylim([0, 40])
plt.title('Day 1')
# plt.savefig('foo.png')
# plt.close(fig)
filename = '/data/Clustering/SleepDeprivation/RatJ/Day2/RatJ_2019-06-02_03-59-19/experiment1/recording1/continuous/Rhythm_FPGA-100.0/continuous.eeg'

SampFreq = 1250
# frames = RecInfo['behavFrames']
# behav = RecInfo['behav']
nChans = 67
ReqChan = 61

# offsetP = ((nREMPeriod - behav[2, 0]) // 1e6) * SampFreq + \
#    int(np.diff(frames[0, :])) + int(np.diff(frames[1, :]))

offsetp = (ReqChan - 1)

b1 = np.memmap(filename, dtype='int16', mode='r',
               offset=1 * (ReqChan - 1) * 2, shape=(1, nChans * SampFreq * 3600 * 15))
eegnrem1 = b1[0, ::nChans]
#eegnrem1 = eegnrem1[0::24]
#sos = sg.butter(3, 100, btype='low', fs=SampFreq, output='sos')
#yf = sg.sosfilt(sos, eegnrem1)
#yf = ft.fft(yf) / len(eegnrem1)
#xf = np.linspace(0.0, SampFreq / 2, len(eegnrem1) // 2)
#y1 = 2.0 / (len(xf)) * np.abs(yf[:len(eegnrem1) // 2])
#y1 = smth.gaussian_filter(y1, 8)

f, t, Sxx = sg.spectrogram(
    eegnrem1, SampFreq, nperseg=1250 * 6, noverlap=1250 * 5, nfft=8000)
Sxx = stat.zscore(Sxx)
Sxx = smth.gaussian_filter(Sxx, sigma=1.5)

plt.subplot(2, 1, 2)
plt.pcolormesh(t / 3600, f, Sxx, cmap='copper', vmax=30)
plt.ylim([0, 40])
plt.xlabel('Time (h)')
plt.ylabel('Frequency (Hz)')
plt.title('Day 2')

plt.pcolormesh(t/3#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 16:32:41 2019

@author: bapung
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack as ft
import scipy.ndimage as smth
import scipy.signal as sg
import scipy.stats as stat
import seaborn as sns

filename = '/data/Clustering/SleepDeprivation/RatJ/Day1/RatJ_2019-05-31_03-55-36/experiment1/recording1/continuous/Rhythm_FPGA-100.0/continuous.eeg'

SampFreq = 1250
#frames = RecInfo['behavFrames']
#behav = RecInfo['behav']
nChans = 75
ReqChan = 28

# offsetP = ((nREMPeriod - behav[2, 0]) // 1e6) * SampFreq + \
#    int(np.diff(frames[0, :])) + int(np.diff(frames[1, :]))

offsetp = (ReqChan-1)

b1 = np.memmap(filename, dtype='int16', mode='r',
               offset=1 * (ReqChan - 1) * 2, shape=(1, nChans * SampFreq * 3600 * 15))
eegnrem1 = b1[0, ::nChans]
# eegnrem1 = eegnrem1[0::24]
# sos = sg.butter(3, 100, btype='low', fs=SampFreq, output='sos')
# yf = sg.sosfilt(sos, eegnrem1)
# yf = ft.fft(yf) / len(eegnrem1)
# xf = np.linspace(0.0, SampFreq / 2, len(eegnrem1) // 2)
# y1 = 2.0 / (len(xf)) * np.abs(yf[:len(eegnrem1) // 2])
# y1 = smth.gaussian_filter(y1, 8)

f, t, Sxx = sg.spectrogram(
    eegnrem1, SampFreq, nperseg=1250 * 6, noverlap=1250 * 5, nfft=8000)

Sxx = stat.zscore(Sxx)
Sxx = smth.gaussian_filter(Sxx, sigma=1.5)

fig = plt.figure()
plt.subplot(2, 1, 1)
# sns.heatmap(Sxx)
plt.pcolormesh(t/3600, f, Sxx, cmap='copper', vmax=30)
plt.ylim([0, 40])
plt.title('Day 1')
# plt.savefig('foo.png')
# plt.close(fig)
filename = '/data/Clustering/SleepDeprivation/RatJ/Day2/RatJ_2019-06-02_03-59-19/experiment1/recording1/continuous/Rhythm_FPGA-100.0/continuous.eeg'

SampFreq = 1250
# frames = RecInfo['behavFrames']
# behav = RecInfo['behav']
nChans = 67
ReqChan = 61

# offsetP = ((nREMPeriod - behav[2, 0]) // 1e6) * SampFreq + \
#    int(np.diff(frames[0, :])) + int(np.diff(frames[1, :]))

offsetp = (ReqChan - 1)

b1 = np.memmap(filename, dtype='int16', mode='r',
               offset=1 * (ReqChan - 1) * 2, shape=(1, nChans * SampFreq * 3600 * 15))
eegnrem1 = b1[0, ::nChans]
#eegnrem1 = eegnrem1[0::24]
#sos = sg.butter(3, 100, btype='low', fs=SampFreq, output='sos')
#yf = sg.sosfilt(sos, eegnrem1)
#yf = ft.fft(yf) / len(eegnrem1)
#xf = np.linspace(0.0, SampFreq / 2, len(eegnrem1) // 2)
#y1 = 2.0 / (len(xf)) * np.abs(yf[:len(eegnrem1) // 2])
#y1 = smth.gaussian_filter(y1, 8)

f, t, Sxx = sg.spectrogram(
    eegnrem1, SampFreq, nperseg=1250 * 6, noverlap=1250 * 5, nfft=8000)
Sxx = stat.zscore(Sxx)
Sxx = smth.gaussian_filter(Sxx, sigma=1.5)

plt.subplot(2, 1, 2)
plt.pcolormesh(t / 3600, f, Sxx, cmap='copper', vmax=30)
plt.ylim([0, 40])
plt.xlabel('Time (h)')
plt.ylabel('Frequency (Hz)')
plt.title('Day 2')

plt.ylim([0, 40])
plt.title('Day 1')#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 16:32:41 2019

@author: bapung
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack as ft
import scipy.ndimage as smth
import scipy.signal as sg
import scipy.stats as stat
import seaborn as sns

filename = '/data/Clustering/SleepDeprivation/RatJ/Day1/RatJ_2019-05-31_03-55-36/experiment1/recording1/continuous/Rhythm_FPGA-100.0/continuous.eeg'

SampFreq = 1250
#frames = RecInfo['behavFrames']
#behav = RecInfo['behav']
nChans = 75
ReqChan = 28

# offsetP = ((nREMPeriod - behav[2, 0]) // 1e6) * SampFreq + \
#    int(np.diff(frames[0, :])) + int(np.diff(frames[1, :]))

offsetp = (ReqChan-1)

b1 = np.memmap(filename, dtype='int16', mode='r',
               offset=1 * (ReqChan - 1) * 2, shape=(1, nChans * SampFreq * 3600 * 15))
eegnrem1 = b1[0, ::nChans]
# eegnrem1 = eegnrem1[0::24]
# sos = sg.butter(3, 100, btype='low', fs=SampFreq, output='sos')
# yf = sg.sosfilt(sos, eegnrem1)
# yf = ft.fft(yf) / len(eegnrem1)
# xf = np.linspace(0.0, SampFreq / 2, len(eegnrem1) // 2)
# y1 = 2.0 / (len(xf)) * np.abs(yf[:len(eegnrem1) // 2])
# y1 = smth.gaussian_filter(y1, 8)

f, t, Sxx = sg.spectrogram(
    eegnrem1, SampFreq, nperseg=1250 * 6, noverlap=1250 * 5, nfft=8000)

Sxx = stat.zscore(Sxx)
Sxx = smth.gaussian_filter(Sxx, sigma=1.5)

fig = plt.figure()
plt.subplot(2, 1, 1)
# sns.heatmap(Sxx)
plt.pcolormesh(t/3600, f, Sxx, cmap='copper', vmax=30)
plt.ylim([0, 40])
plt.title('Day 1')
# plt.savefig('foo.png')
# plt.close(fig)
filename = '/data/Clustering/SleepDeprivation/RatJ/Day2/RatJ_2019-06-02_03-59-19/experiment1/recording1/continuous/Rhythm_FPGA-100.0/continuous.eeg'

SampFreq = 1250
# frames = RecInfo['behavFrames']
# behav = RecInfo['behav']
nChans = 67
ReqChan = 61

# offsetP = ((nREMPeriod - behav[2, 0]) // 1e6) * SampFreq + \
#    int(np.diff(frames[0, :])) + int(np.diff(frames[1, :]))

offsetp = (ReqChan - 1)

b1 = np.memmap(filename, dtype='int16', mode='r',
               offset=1 * (ReqChan - 1) * 2, shape=(1, nChans * SampFreq * 3600 * 15))
eegnrem1 = b1[0, ::nChans]
#eegnrem1 = eegnrem1[0::24]
#sos = sg.butter(3, 100, btype='low', fs=SampFreq, output='sos')
#yf = sg.sosfilt(sos, eegnrem1)
#yf = ft.fft(yf) / len(eegnrem1)
#xf = np.linspace(0.0, SampFreq / 2, len(eegnrem1) // 2)
#y1 = 2.0 / (len(xf)) * np.abs(yf[:len(eegnrem1) // 2])
#y1 = smth.gaussian_filter(y1, 8)

f, t, Sxx = sg.spectrogram(
    eegnrem1, SampFreq, nperseg=1250 * 6, noverlap=1250 * 5, nfft=8000)
Sxx = stat.zscore(Sxx)
Sxx = smth.gaussian_filter(Sxx, sigma=1.5)

plt.subplot(2, 1, 2)
plt.pcolormesh(t / 3600, f, Sxx, cmap='copper', vmax=30)
plt.ylim([0, 40])
plt.xlabel('Time (h)')
plt.ylabel('Frequency (Hz)')
plt.title('Day 2')

# plt.savefig('foo.png')
# plt.close(fig)
filename = '/data/Clustering/SleepDeprivation/RatJ/Day2/RatJ_2019-06-02_03-59-19/experiment1/recording1/continuous/Rhythm_FPGA-100.0/continuous.eeg'

SampFreq = 1250
# frames = RecInfo['behavFrames']
# behav = RecInfo['behav']
nChans = 67
ReqChan = 61

# offsetP = ((nREMPeriod - behav[2, 0]) // 1e6) * SampFreq + \
#    int(np.diff(frames[0, :])) + int(np.diff(frames[1, :]))

offsetp = (ReqChan - 1)

b1 = np.memmap(filename, dtype='int16', mode='r',
               offset=1 * (ReqChan - 1) * 2, shape=(1, nChans * SampFreq * 3600 * 15))
eegnrem1 = b1[0, ::nChans]
#eegnrem1 = eegnrem1[0::24]
#sos = sg.butter(3, 100, btype='low', fs=SampFreq, output='sos')
#yf = sg.sosfilt(sos, eegnrem1)
#yf = ft.fft(yf) / len(eegnrem1)
#xf = np.linspace(0.0, SampFreq / 2, len(eegnrem1) // 2)
#y1 = 2.0 / (len(xf)) * np.abs(yf[:len(eegnrem1) // 2])
#y1 = smth.gaussian_filter(y1, 8)

f, t, Sxx = sg.spectrogram(
    eegnrem1, SampFreq, nperseg=1250 * 6, noverlap=1250 * 5, nfft=8000)
Sxx = stat.zscore(Sxx)
Sxx = smth.gaussian_filter(Sxx, sigma=1.5)

plt.subplot(2, 1, 2)
plt.pcolormesh(t / 3600, f, Sxx, cmap='copper', vmax=30)
plt.ylim([0, 40])
plt.xlabel('Time (h)')
plt.ylabel('Frequency (Hz)')
plt.title('Day 2')
