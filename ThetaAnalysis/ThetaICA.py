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



sourceDir = '/data/DataGen/wake_new/'

arrays = {}
f= h5py.File(sourceDir + 'wake-basics.mat', 'r') 
for k, v in f.items():
    arrays[k] = np.array(v)

#spks = {}
spikes= h5py.File(sourceDir + 'testVersion.mat', 'r') 
#for k, v in f1.items():
#    spks[k] = np.array(v)
    
subjects = arrays['basics']
#spikes = spks['spikes']


a1 = np.array(spikes['spikes']['KevinMaze1']['time'])
for sub in range(0,1):
    sub_name = subjects[sub]
    print(sub_name)
    
    celltype = spikes['spikes'][sub_name]['time']
    
    print(len(celltype))







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