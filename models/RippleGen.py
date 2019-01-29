# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 22:20:25 2019

@author: Bapun
"""

import numpy as np

gL = 7
E_L= -58
Delta = 2
Vt = -50
Vrest = -46 
Vthresh = 0
alpha = 2
b = 40

v= np.zeros(100)
w = np.zeros(100)





for i in range(0,100):
    v[i+1] = v[i] + -gL(v[i]-E_L) + gL*Delta*np.exp((v[i]-Vt)/Delta)- W + I[i]
            
    w[i+1] = alpha*(v[i]- E_L)-w
    
    I[i+1]= I_DC + beta*eta_t + Isyn[i]
    
    
    
    if v[i+1] > Vthresh:
        v[i+1]= Vrest
        w[i+1] = w[i]+b
        
    