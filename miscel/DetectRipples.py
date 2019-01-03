#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 15:26:23 2019

@author: bapung
"""

def DetectRipples():
    zscSig = stats.zscore(filteredSig)
    envSig = hilbert(zscSig)
    amplitude_envelope = np.abs(envSig)
    
    rip = np.diff(np.where(amplitude_envelope > Thresh,1,0)) 
    rip_begin = np.where(rip>0)
    rip_end = np.where(rip<0)
    
    rip_dur = rip_end[0]-rip_begin[0]
    
    rip_all= np.array([rip_begin[0] , rip_end[0]]).transpose()
    interDist = 50*1000/1250
    minRipSize =  30*1000/1250
    maxRipSize = 450*1000/1250
    
    ripple = rip_all[0,:]
    mrgRipple = []
    for i in range(0,len(rip_all)):
        if rip_all[i,0]-ripple[1] < interDist:
            ripple = [ripple[0],rip_all[i,1]]
        else:
            mrgRipple.append(ripple)
            ripple = rip_all[i,:]
                
    ripple_merged = np.array(mrgRipple)
    
    ripl_peak = []
    for i1 in range(0, len(ripple_merged)):
        if np.max(amplitude_envelope[ripple_merged[i1,0]:ripple_merged[i1,1]])>5:
            ripl_peak.append(ripple_merged[i1,:])
    
    ripl_peak = np.array(ripl_peak)
    
    RipDur= ripl_peak[:,1]-ripl_peak[:,0]
    DiscardRip = np.where((RipDur < minRipSize) | (RipDur > maxRipSize))[0]
     
    ripl_peak = np.delete(ripl_peak,DiscardRip, axis=0)