#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 11:27:45 2019

@author: bapung
"""
#Comparing spectral profile from first NREM episode to last NREM episode
import numpy as np
import matplotlib.pyplot as plt
#import scipy.fftpack as sfft
#import time
import scipy.io as sio
import scipy.ndimage.filters as smth
import scipy.fftpack as ft
import scipy.stats as stat
import scipy.signal as sg
from SpectralAnalysis import lfpSpect
import h5py



sourceDir = '/data/DataGen/wake_new/'
sourceDir2 = '/data/DataGen/sleep/'

arrays = {}
f= h5py.File(sourceDir + 'wake-basics.mat', 'r') 
for k, v in f.items():
    arrays[k] = np.array(v)

fspikes= h5py.File(sourceDir + 'testVersion.mat', 'r') 
fbehav= h5py.File(sourceDir + 'wake-behavior.mat', 'r') 
slpbehav= sio.loadmat(sourceDir2 + 'sleep-behavior.mat') 

lastNrem = np.array([1.26586295e+11, 1.27128997e+11]).reshape(1,2)
    
subjects = arrays['basics']


for sub in range(1,2):
    sub_name = subjects[sub]
    print(sub_name)
    
    nUnits = len(fspikes['spikes'][sub_name]['time'])
    celltype={}
    quality={}
    stability={}
    for i in range(0,nUnits):
        celltype[i] = fspikes[fspikes['spikes'][sub_name]['time'][i,0]].value
        quality[i] = fspikes[fspikes['spikes'][sub_name]['quality'][i,0]].value
        stability[i] = fspikes[fspikes['spikes'][sub_name]['StablePrePost'][i,0]].value
    
    behav = np.transpose(fbehav['behavior'][sub_name]['time'][:])
    states = np.transpose(fbehav['behavior'][sub_name]['list'][:])
    frames = np.transpose(fbehav['behavior'][sub_name]['eegFrame'][:])
    pyrid = [i for i in range(0,nUnits) if quality[i] < 4 and stability[i] == 1]
    cellpyr= [celltype[a] for a in pyrid]
    
    
    
    BasicInfo = {'samplingFrequency': 1250}
    BasicInfo['behavFrames'] = frames
    BasicInfo['behav'] = behav
    BasicInfo['numChannels'] = 65
    BasicInfo['SpectralChannel'] = 50
    
    nMazeFrames = int(np.diff(frames[2,:]))
    POSTNREM = states[(states[:,0]>behav[2,0]) & (states[:,2]==1),:]
    
    
    y1,xf = lfpSpect(sub_name,POSTNREM[15,0],BasicInfo)
    y2,xL = lfpSpect(sub_name,lastNrem[0,0],BasicInfo)

#    fig, ax = plt.subplots()
    plt.clf()
    ax0 = plt.subplot(1,1,1)
    plt.plot(xf, y1)
    plt.plot(xL, y2,'r')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (db)')
    plt.yscale('log')
    plt.xlim(0.5,100)
    ax0.spines['right'].set_visible(False)
    ax0.spines['top'].set_visible(False)
        

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    
