# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 23:23:37 2019

@author: Bapun
"""

import numpy as np
import matplotlib.pyplot as plt




V_reset = -0.080;  
V_e = -0.075
V_th = -0.040
Rm = 10e6
tau_m = 10e-3


dt = 0.0002
T = np.arange(0,1,dt)

Vm= np.zeros((len(T),1))
Vm[0] = V_reset;
Im = 5e-9;

for t in range (len(T)-1):
    if Vm[t] > V_th:
        Vm[t+1] = V_reset
    else:
        Vm[t+1] = Vm[t] + dt * ( -(Vm[t] - V_e) + Im * Rm) / tau_m



plt.plot(T,Vm,'b-');

