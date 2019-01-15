#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 11:24:29 2019

@author: bapung
"""

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

dt = 0.001
Tmax = 2
t = np.arange(0,Tmax, dt)
fs = len(t)/Tmax
noise_power = 0.01 * fs / 2
sin = lambda x,p: np.sin(2*np.pi*x*t + p)

y1 = 5*sin(30,0)
y1 += np.random.normal(scale=np.sqrt(noise_power), size=t.shape)
f, tx, Sxx = signal.spectrogram(y1, fs=1000,nperseg=2*100, noverlap=20)

#fs = 10e3
#N = 1e5
#amp = 2 * np.sqrt(2)
#noise_power = 0.01 * fs / 2
#time = np.arange(N) / float(fs)
#mod = 500*np.cos(2*np.pi*0.25*time)
#carrier = amp * np.sin(2*np.pi*3e3*time + mod)
#noise = np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
#noise *= np.exp(-time/5)
#x = carrier + noise
#
#f, t, Sxx = signal.spectrogram(x, fs)
#plt.pcolormesh(t, f, Sxx)
#plt.ylabel('Frequency [Hz]')
#plt.xlabel('Time [sec]')
#plt.show()

plt.clf()
plt.subplot(2,1,1)
plt.plot(t,y1)
plt.subplot(2,1,2)
plt.pcolormesh(tx, f, Sxx)

