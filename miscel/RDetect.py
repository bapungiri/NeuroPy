#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 10:35:04 2019

@author: bapung
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sg
import scipy.stats as stats
from scipy.signal import hilbert
import h5py


sourceDir = '/data/DataGen/wake_new/'
fbehav= h5py.File(sourceDir + 'wake-behavior.mat', 'r') 

sub_name = 'RoyMaze1'
behav = np.transpose(fbehav['behavior'][sub_name]['time'][:])
states = np.transpose(fbehav['behavior'][sub_name]['list'][:])
frames = np.transpose(fbehav['behavior'][sub_name]['eegFrame'][:])

ThetaChannel = 50
nMazeFrames = int(np.diff(frames[0,:]))
Thresh = 2

b = np.memmap('/data/EEGData/' + sub_name + '.eeg', dtype='int16', mode='r', offset=int(frames[0,0]*65*2+1*(ThetaChannel-1)*2)
                ,shape=(1,65*nMazeFrames))
eegMaze = b[0,:]
eegMaze = eegMaze[ThetaChannel-1::65]


testData = eegMaze[:1000000]
sos = sg.butter(3, [140, 250], 'bandpass', fs=1250,output='sos')
filteredSig = sg.sosfilt(sos, testData)




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
for i in range(len(rip_all)):
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

alltog = np.concatenate([testData[x:y] for x,y in zip(ripl_peak[:,0], ripl_peak[:,1])], axis=0)

nRow = 4

plt.clf()
plt.subplot(nRow, 1, 1)
plt.plot(testData)
plt.subplot(nRow, 1, 2)
plt.plot(stats.zscore(np.square(filteredSig)))
plt.plot(amplitude_envelope)
plt.subplot(nRow, 1, 3)
plt.plot(alltog)