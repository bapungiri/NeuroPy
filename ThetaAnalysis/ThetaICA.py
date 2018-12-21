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
import scipy.stats as stat
import scipy.signal as sg
import h5py
#import tables
#import struct
from PyEMD import EEMD

eemd = EEMD()
emd = eemd.EMD
emd.extrema_detection="parabol"


sourceDir = '/data/DataGen/wake_new/'

arrays = {}
f= h5py.File(sourceDir + 'wake-basics.mat', 'r') 
for k, v in f.items():
    arrays[k] = np.array(v)

#spks = {}
fspikes= h5py.File(sourceDir + 'testVersion.mat', 'r') 
fbehav= h5py.File(sourceDir + 'wake-behavior.mat', 'r') 
fICAStrength = h5py.File('/data/DataGen/ICAStrengthpy.mat', 'r') 
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
    states = np.transpose(fbehav['behavior'][sub_name]['list'][:])
    frames = np.transpose(fbehav['behavior'][sub_name]['eegFrame'][:])
    ICA_strength = np.array(np.transpose(fICAStrength['ActStrength']['subjects'][sub_name]['wake'][:]))
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

    
    f, t, Sxx = sg.spectrogram(eegMaze,fs=1250,nperseg=2*1250, noverlap=400) 
    
    tICA = np.linspace(0,np.diff(behav[2,:])/1e6,np.shape(ICA_strength)[1])
    ica_zsc = stat.zscore(ICA_strength,axis=1)
    ica_zsc_= ica_zsc < 3
    ica_zsc[ica_zsc_]=0
    ica_zsc_= ica_zsc >= 3
    ica_zsc[ica_zsc_]=1
    
    wakeID = [i for i in range(0,len(states)) if states[i,0] > behav[1,0] and states[i,0]<behav[1,1] and states[i,2]==4]
#    wakeTheta = states[states[:,0]>behav[1,0] and states[:,0]<behav[1,1]]
    wakeTheta = states[wakeID,:]
    
    ThetaDur = np.diff(wakeTheta[:,[0,1]],axis=1)/1e6
    
    
    firstTheta = (wakeTheta[0,[0,1]] - behav[1,0])/1e6
    firstTheta = np.transpose([378,381])
    eegfirstFrame = firstTheta*1250
    eegfirstepoch = eegMaze[int(eegfirstFrame[0]):int(eegfirstFrame[1])]
    ThetaTime = np.linspace(0,3,len(eegfirstepoch))
    
   
    
    eIMFs = eemd.eemd(eegfirstepoch, ThetaTime)
    nIMFs = eIMFs.shape[0]
    
    plt.clf()
    plt.subplot(nIMFs+1, 1, 1)
    plt.plot(ThetaTime,eegfirstepoch)
    
    
    
    for n in range(nIMFs):
        plt.subplot(nIMFs+1, 1, n+2)
        plt.plot(ThetaTime, eIMFs[n], 'g')
        plt.ylabel("eIMF %i" %(n+1))
        plt.locator_params(axis='y', nbins=5)
    
    
    
#    emd
    
#    t1 = np.linspace(0,7.2,9000-1)
#    eIMFs = eemd.eemd(eegMaze[1:9000], t1, max_imf=5)
#    nIMFs = eIMFs.shape[0]
    
#    plt.clf()
#    plt.subplot(4, 1, 1)
#    plt.pcolormesh(t, f[0:200], Sxx[1:200,:],cmap = plt.cm.get_cmap("hot"), vmin=-400, vmax=950000)
##    plt.imshow(Sxx)
#    plt.ylabel('Frequency [Hz]')
#    plt.xlabel('Time [sec]')
##    plt.show()
#    plt.subplot(4, 1, 2)
#    plt.pcolormesh(tICA, np.linspace(1,20,20),ica_zsc,cmap = plt.cm.get_cmap("hot"))
##
#    for n in range(4):
#        plt.subplot(4, 1, n+2)
#        plt.plot(t1, eIMFs[n], 'g')
#        plt.ylabel("eIMF %i" %(n+1))
#        plt.locator_params(axis='y', nbins=5)
    
#    plt.plot(eegMaze[1:2000])

#    
    
    
