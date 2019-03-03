# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 22:39:44 2018

@author: Bapun
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sg
from OsCheck import DataDirPath


t = np.linspace(0, 100, 20000)

c = [3]

b = DataDirPath()


def sin(f, p): return np.sin(2 * np.pi * f * t + p)


y = sin(10, 0)

y += sin(20, 0)

a = 0
for i in [1, 3, 4]:
    a = a + 2
    print(a)


spec = sg.spectrogram(y, 1 / 200)
print('Hello World')

plt.plot(t, y)
plt.show()
