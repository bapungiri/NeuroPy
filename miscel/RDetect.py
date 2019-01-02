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


testData = eegMaze[:10000]
sos = sg.butter(3, [140/1250, 250/1250], 'bandpass', output='sos')
filteredSig = sg.sosfilt(sos, testData)

zscSig = stats.zscore(filteredSig)
rip = np.diff(np.where(zscSig > Thresh,1,0) )
rip_begin = np.where(rip>0)
rip_end = np.where(rip<0)

rip_dur = rip_end-rip_begin


nRow = 3

plt.clf()
plt.subplot(nRow, 1, 1)
plt.plot(testData)
plt.subplot(nRow, 1, 2)
plt.plot(filteredSig)