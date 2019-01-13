# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 23:23:37 2019

@author: Bapun
"""

import numpy as np
import matplotlib.pyplot as plt



Itheta = 0.065*40
V_reset = -0.080;  
E0 = -0.075
V_th = -0.040
Rm = 10e6
tau_m = 10e-3
Ic = 80
t_restI = 0
t_restE=0


dt = 0.0002
T = np.arange(0,1,dt)
I0 = np.zeros((len(T),1))
pos = T*40/1000
Ie0 = 110.0+0.5*40.0
Vm= np.zeros((len(T),1))
Vm[0] = V_reset;
Im = 5e-9;

for t in range (len(T)-1):
    
    I0[t+1] = Ic - Itheta*np.cos(2.0*np.pi*0.008*T[t])
    Iext[t+1]=Ie0*np.exp(-(abs(pos[t]-200))**2/(2*40^2))
    g[t+1]= -g[t]/tauI + w_j 
    if T[i] > t_restI:
        g[i]=g[i-1]+(-g[i-1]/tauI+val)*dt
        eta = np.random.randn()
        v[i]= v[i-1]+ (- ((v[i-1]-E0)/taum) - (g[i-1]*(v[i-1]-EI)/Cm) + (I0[i-1]/Cm)+(eta*0.01/sqrt(taum)))*dt
        val=0.0
        
    if time[i] > t_restE:
        ge[i]=ge[i-1]+(-ge[i-1]/tauE+val)*dt
        eta = randn();
        ve[i]= ve[i-1]+ (- ((ve[i-1]-E0)/taumE) - (ge[i-1]*(ve[i-1]-Ee)/CmE) + (Ie[i-1]/CmE)+(eta*0.7/sqrt(taumE)))*dt
        vale=0.0     
        
    if Vm[t] > V_th:
        Vm[t+1] = V_reset
        
    else:
        Vm[t+1] = Vm[t] +  (-(Vm[t] - E0) / tau_m + Iext/Cm + sigma_n*eta/sqrt(tau_m)
                    - g[t]*(Vm[t]-EI)/Cm)* dt



plt.plot(T,Vm,'b-');

