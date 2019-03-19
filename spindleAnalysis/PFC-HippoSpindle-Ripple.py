#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 09:48:15 2019

@author: bapung
"""


import numpy as np
import pandas as pd
#from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
#from scipy.ndimage import gaussian_filter
from OsCheck import DataDirPath, figDirPath
#import scipy.signal as sg
#import scipy.stats as stats
#from scipy.signal import hilbert
import h5py
import seaborn as sns
#sns.set(style="darkgrid")


#plt.style.use('seaborn')
        

sourceDir = DataDirPath() + 'sleep/'
figFilename = figDirPath() +'SpindleAnalysis/PFCSpindle-HippRipple.pdf'

arrays = {}
f= h5py.File(sourceDir + 'sleep-basics.mat', 'r') 
for k, v in f.items():
    arrays[k] = np.array(v)

# spks = {}
# fspikes= h5py.File(sourceDir + 'sleep-spikes.mat', 'r') 
fbehav= h5py.File(sourceDir + 'sleep-behavior.mat', 'r') 
fripple = h5py.File(sourceDir + 'sleep-ripple.mat', 'r') 
fspindle= h5py.File(sourceDir + 'sleep-spindle.mat', 'r') 
#fICAStrength = h5py.File('/data/DataGen/ICAStrengthpy.mat', 'r') 

subjects = arrays['basics']
#spikes = spks['spikes']

#figFilename = figDirPath() +'PlaceCells.pdf'
#pdf = PdfPages(figFilename)
k=1
m=[]
plt.clf()
#f, axarr = plt.subplots(2, 4)

for sub in range(0,19):
    sub_name = subjects[sub]
    print(sub_name)
    
    area = list((fspindle['spindle'][sub_name]).keys())
    
    if len(area)>1 :
        
        pfcSpindle = np.transpose(fspindle['spindle'][sub_name]['CTX']['peakTime'][:])
#        hpcSpindle = np.transpose(fspindle['spindle'][sub_name]['HPC']['peakTime'][:])
        hpcRipple = np.transpose(fripple['ripple'][sub_name]['peakTime'][:])
        
        diffTime = ((pfcSpindle.transpose() - hpcRipple.transpose()).ravel())/1e6
#        diffTime = ((hpcRipple - pfcSpindle).ravel())/1e6
        if 'Steve' in sub_name:
            
            if 'Sleep' in sub_name:
            
        
                m.append(plt.subplot(1,2,1))
                hist, edge = np.histogram(diffTime, bins= np.arange(-30,30,1))
                plt.plot(edge[0:len(hist)], hist, color='#f49e42')
                plt.title(sub_name)
            
            else:
                m.append(plt.subplot(1,2,1))
                hist, edge = np.histogram(diffTime, bins= np.arange(-30,30,1))
                plt.plot(edge[0:len(hist)], hist, color = '#726f6b')
                plt.title(sub_name)
                
                
            
        if 'Ted' in sub_name:
            
        
            if 'Sleep' in sub_name:
            
        
                m.append(plt.subplot(1,2,2))
                hist, edge = np.histogram(diffTime, bins= np.arange(-30,30,1))
                plt.plot(edge[0:len(hist)], hist, color='#f49e42')
                plt.title(sub_name)
            
            else:
                m.append(plt.subplot(1,2,2))
                hist, edge = np.histogram(diffTime, bins= np.arange(-30,30,1))
                plt.plot(edge[0:len(hist)], hist, color = '#726f6b')
                plt.title(sub_name)

for pltind in [4,5,6,7]:
    m[pltind].set(xlabel='Time relative to PFC spindle (s)')

for pltind in [0,4]:
    m[pltind].set(ylabel='# Ripple events')      
     
     
plt.savefig(figFilename, dpi = 300)
    