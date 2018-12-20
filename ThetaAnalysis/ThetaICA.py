#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 13:36:34 2018

@author: bapung
"""



import numpy as np
import matplotlib.pyplot as plt
#import scipy.fftpack as sfft
#import time
import scipy.io as sio
import h5py
#import tables
#import struct


sourceDir = '/data/DataGen/wake_new/'

arrays = {}
f= h5py.File(sourceDir + 'wake-basics.mat', 'r') 
for k, v in f.items():
    arrays[k] = np.array(v)

#spks = {}
fspikes= h5py.File(sourceDir + 'testVersion.mat', 'r') 
fbehav= h5py.File(sourceDir + 'wake-behavior.mat', 'r') 
#for k, v in f1.items():
#    spks[k] = np.array(v)
    
subjects = arrays['basics']
#spikes = spks['spikes']


#a1 = np.array(spikes['spikes']['KevinMaze1']['time'][0])
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
    frames = np.transpose(fbehav['behavior'][sub_name]['eegFrame'][:])
    pyrid = [i for i in range(0,nUnits) if quality[i] < 4 and stability[i] == 1]
    cellpyr= [celltype[a] for a in pyrid]
    
    ThetaChannel = 50
    
#    fid = open('/data/EEGData/' + sub_name + '.eeg', 'rb')
#    fid.seek(int(frames[2,1]*65*2+1*(ThetaChannel-1)*2),0)
    
    nMazeFrames = int(np.diff(frames[2,:]))
    b = np.memmap('/data/EEGData/' + sub_name + '.eeg', dtype='int16', mode='r', offset=int(frames[2,1]*65*2+1*(ThetaChannel-1)*2)
                ,shape=(1,65*nMazeFrames))
    eegMaze = b[0,:]
    eegMaze = eegMaze[ThetaChannel-1::65]
#    print a[1,5]
#    fid = open('/data/EEGData/' + sub_name + '.eeg', 'rb')
#    dim = np.fromfile(fid, dtype='>u4')
#    fid.seek(int(frames[2,1]*65*2+1*(ThetaChannel-1)*2),0)
##    flCont = fid.read(30)
##    fg = struct.unpack(fid,4)
#    
#    Maze = []
#    nMazeFrames = np.diff(frames[2,:])
#    checkE  = np.fromfile(fid, dtype='int16', count=int(nMazeFrames))
#    plt.clf()
#    while True:
#        Maze = np.append(Maze,np.fromfile(fid, dtype='int16', count=1))
#        fid.seek((65-1)*2,1)
#        if len(Maze)>nMazeFrames: break
    plt.clf()
    plt.plot(eegMaze[1:2000])

#    
    
    
