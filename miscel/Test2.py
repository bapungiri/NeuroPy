# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 22:39:44 2018

@author: Bapun
"""

import numpy as np
import scipy.signal as sg


t = np.linspace(0,100,20000)
sin = lambda f,p: np.sin(2*np.pi*f*t+p)

y = sin(10,0)
y += sin(20,0)

spec = sg.spectrogram(y,1/200, )
