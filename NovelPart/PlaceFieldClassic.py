#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 14:00:27 2019

@author: bapung
"""
import numpy as np
from scipy.ndimage import gaussian_filter

def PlaceFieldClassic(Pos, Spkt, Smooth, nGrid):
    
    
        
    xcoord = np.arange(min(Pos[0,:]),max(Pos[0,:])+1,2)
    ycoord = np.arange(min(Pos[1,:]),max(Pos[1,:])+1,2)
    spktx = np.interp(Spkt,Pos[2,:],Pos[0,:])
    spkty = np.interp(Spkt,Pos[2,:],Pos[1,:])
        
    pf, xe, ye = np.histogram2d(spktx,spkty,bins = [xcoord,ycoord])
    pft = pf*(1/30)
    
    eps = np.spacing(1)
    pfRate = pf/(pft+eps)
    
    pfRate_smooth= gaussian_filter(pfRate, sigma=2)
    
    return pfRate_smooth
