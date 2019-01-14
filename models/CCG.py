#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 16:28:33 2019

@author: bapung
"""

import numpy as np
import numpy.matlib as mlib

def spikegram(x,y):
    
    neur1 =  mlib.repmat(x,len(y),1)
    neur2 = mlib.repmat(y,len(x),1)
    
    lenx = np.shape(x)[1]
    leny = np.shape(y)[1]
    
#     diff1 = (neur1-mlib.repmat(np.transpose(y),1,len(x))).reshape(1,lenx*leny)
#     diff2 = (neur2-mlib.repmat(np.transpose(x),1,len(y))).reshape(1,lenx*leny)
    
    diff1 = neur1-neur2[:,np.newaxis]
    diff2 = (neur2-mlib.repmat(np.transpose(x),1,len(y))).reshape(1,lenx*leny)
    
    tbin = np.linspace(-5,5,100) 
    corr1, bedge = np.histogram(diff1,tbin) 
    return corr1