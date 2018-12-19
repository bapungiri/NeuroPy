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
import struct


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
    pyrid = [i for i in range(0,nUnits) if quality[i] < 4 and stability[i] == 1]
    cellpyr= [celltype[a] for a in pyrid]
    
    ThetaChannel = 50
    fid = open('/data/EEGData/' + sub_name + '.eeg', 'rb')
    dim = np.fromfile(fid, dtype='>u4')
    fid.seek(1*(ThetaChannel-1)*2,0)
#    flCont = fid.read(30)
#    fg = struct.unpack(fid,4)
    
    fe = []
    
    while True:
        fe = np.append(fe,np.fromfile(fid, dtype='int16', count=1))
        fid.seek((65-1)*2,1)
        if len(fe)>1000: break
    
    plt.plot(fe)

#       floats = []
#    with open('/data/EEGData/' + sub_name + '.eeg', 'rb') as f:
#        while True:
#            buff = f.read(4)                # 'f' is 4-bytes wide
#            if len(x) > 20: break
#            x = struct.unpack('f', buff)[0] # Convert buffer to float (get from returned tuple)
#            floats.append(x)                # Add float to list (for example)
#            f.seek(8, 0)    
    
#    fseek(fh,1*(ThetaChannel-1)*2,'bof');
#    EEGPRE=fread(fh,[1,frames(1,2)-frames(1,1)],'int16',(NumChan-1)*2);
#   
#    REM_pre = states((states(:,1)<behav(1,2) & states(:,3)==2),1:2)';
#    
#    InputParam.ThetaStates = REM_pre(:)';
#    InputParam.timePoints = linspace(behav(1,1),behav(1,2),length(EEGPRE));
#    InputParam.EEG = EEGPRE;
#    for cellid in range(0, len(cellpyr)):
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#    d = {value: foo(value) for value in sequence if bar(value)}
#    fr = np.histogram(celltype)
    
#    plt.plot(fr[0])







#fle1 = tables.open_file(sourceDir + 'wake-basics.mat')
#fg1 = fle1.root.basics
#fg2 = fle1.get_node('/basics/RoyMaze1')[:]


#subnames = [x['Group'] for x in fle1.root.basics]
#fg = sio.loadmat()
#bg = fle1.root.basics[:]
#lat = file.root.lat[:]
## Alternate syntax if the variable name is in a string
#varname = 'basics'
#lon = fle1.get_node('/' + varname)

#f1 =h5py.File(sourceDir + 'wake-spikes.mat','r')
#data1 = f1['spikes'] 
##spikes = np.array(data1)
#
#
#f =h5py.File(sourceDir + 'wake-basics.mat','r')
#data = f.get('basics') 
#data = np.array(data)
#
#
#for sub in range(0, 6):
#    sub_name = data[sub]
#    




#a =sio.loadmat(sourceDir + 'wake-basics.mat')

#start = time.time()


#fid = open('/data/EEGData/RoyMaze1.eeg', 'rb')
#dim = np.fromfile(fid, dtype='>u4')

#end = time.time()
#print(end - start)
#
#N = 600
## sample spacing
#T = 1.0 / 800.0
#x = np.linspace(0.0, N*T, N)
#y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
#yf = sfft.fft(y)
#xf = np.linspace(0.0, 1.0/(2.0*T), N/2)

#fig, ax = plt.subplots()
#ax.plot(xf, 2.0/N * np.abs(yf[:N//2]))
#plt.show()