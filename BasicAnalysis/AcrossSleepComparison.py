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
import scipy.ndimage.filters as smth
import scipy.fftpack as ft
import scipy.stats as stat
import scipy.signal as sg
import h5py
#import tables
#import struct



sourceDir = '/data/DataGen/wake_new/'

arrays = {}
f= h5py.File(sourceDir + 'wake-basics.mat', 'r') 
for k, v in f.items():
    arrays[k] = np.array(v)

fspikes= h5py.File(sourceDir + 'testVersion.mat', 'r') 
fbehav= h5py.File(sourceDir + 'wake-behavior.mat', 'r') 
fICAStrength = h5py.File('/data/DataGen/ICAStrengthpy.mat', 'r') 

    
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
    ICA_strength = np.array(np.transpose(fICAStrength['ActStrength']['subjects'][sub_name]['wake'][:]))
    pyrid = [i for i in range(0,nUnits) if quality[i] < 4 and stability[i] == 1]
    cellpyr= [celltype[a] for a in pyrid]
    
    ThetaChannel = 50
    
#    fid = open('/data/EEGData/' + sub_name + '.eeg', 'rb')
#    fid.seek(int(frames[2,1]*65*2+1*(ThetaChannel-1)*2),0)
    
    nMazeFrames = int(np.diff(frames[2,:]))
    b = np.memmap('/data/EEGData/' + sub_name + '.eeg', dtype='int16', mode='r', offset=int(frames[2,0]*65*2+1*(ThetaChannel-1)*2)
                ,shape=(1,65*nMazeFrames))
    eegMaze = b[0,:]
    eegMaze = eegMaze[ThetaChannel-1::65]
    
    eegTime = np.linspace(behav[2,0],behav[2,1],len(eegMaze))
    
    POSTNREM = states[(states[:,0]>behav[2,0]) & (states[:,2]==1),:]
    
#    a = np.where((eegTime > POSTNREM[0,0]) & (eegTime < POSTNREM[0,1]))
    
    eeg1st = eegMaze[np.where((eegTime > POSTNREM[0,0]) & (eegTime < POSTNREM[0,1]))]
    eegLast = eegMaze[np.where((eegTime > POSTNREM[-1,0]) & (eegTime < POSTNREM[-1,1]))]
    

    
    
    sos = sg.butter(3, 50, btype = 'low', fs=1250, output='sos')
    
    yf = sg.sosfilt(sos,eeg1st)
    yL = sg.sosfilt(sos,eegLast)
    
    yf = ft.fft(yf)/len(eeg1st)
    yL = ft.fft(yL)/len(eegLast)
    
    xf = np.linspace(0.0, 1250/2, len(eeg1st)/2)
    xL = np.linspace(0.0, 1250/2, len(eegLast)/2)
    
    
    y1 = 2.0/(len(xf)/2) * np.abs(yf[:10000])
    y2 = 2.0/(len(xL)/2) * np.abs(yL[:80000])
    
    y1 = smth.gaussian_filter(y1,8)
    y2 = smth.gaussian_filter(y2,8)
    

#    fig, ax = plt.subplots()
    plt.clf()
    plt.plot(xf[:10000], y1)
    plt.plot(xL[:80000], y2,'r')
    plt.yscale('log')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    
