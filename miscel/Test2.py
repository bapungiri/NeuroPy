# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 22:39:44 2018

@author: Bapun
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sg
# import matplotlib as mp


t = np.linspace(0, 100, 20000)


def sin(f, p): return np.sin(2*np.pi*f*t+p)


y = sin(10, 0)

y += sin(20, 0)

spec = sg.spectrogram(y, 1/200)
print('Hello World')

plt.plot(t, y)
