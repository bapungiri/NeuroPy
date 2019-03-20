#!/usr/bin/env python3
#-*- coding: utf-8 -*-


import numpy as np
import pandas as pd
#from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.ndimage.filters import gaussian_filter1d
from OsCheck import DataDirPath, figDirPath
# import scipy.signal as sg
# import scipy.stats as stats
# from scipy.signal import hilbert
import h5py
import seaborn as sns

# sns.set(style="darkgrid")


# plt.style.use('seaborn')
        
colmapDark = plt.cm.Greys(np.linspace(0.35, 1, 3))
colmapLight = plt.cm.Oranges(np.linspace(0.3, 1, 4))

sourceDir = DataDirPath() + 'sleep/'
figFilename = figDirPath() + 'SpindleAnalysis/PFCSpindle-HippSpindle_normSmoothed.pdf'

arrays = {}
f = h5py.File(sourceDir + 'sleep-basics.mat', 'r')
for k, v in f.items():
    arrays[k] = np.array(v)

# spks = {}
# fspikes= h5py.File(sourceDir + 'sleep-spikes.mat', 'r') 
fbehav = h5py.File(sourceDir + 'sleep-behavior.mat', 'r') 
fripple = h5py.File(sourceDir + 'sleep-ripple.mat', 'r') 
fspindle = h5py.File(sourceDir + 'sleep-spindle.mat', 'r') 
# fICAStrength = h5py.File('/data/DataGen/ICAStrengthpy.mat', 'r') 

subjects = arrays['basics']
# spikes = spks['spikes']

# figFilename = figDirPath() +'PlaceCells.pdf'
# pdf = PdfPages(figFilename)
k = 1
m = []
plt.clf()
# f, axarr = plt.subplots(2, 4)

for sub in range(0, 19):
    sub_name = subjects[sub]
    print(sub_name)
    
    area = list((fspindle['spindle'][sub_name]).keys())
    
    if len(area) > 1:
        
        pfcSpindle = np.transpose(fspindle['spindle'][sub_name]['CTX']['peakTime'][:])
        hpcSpindle = np.transpose(fspindle['spindle'][sub_name]['HPC']['peakTime'][:])
#        hpcRipple = np.transpose(fripple['ripple'][sub_name]['peakTime'][:])
        
        diffTime = ((pfcSpindle.transpose() - hpcSpindle).ravel())/1e6
#        diffTime = ((hpcRipple - pfcSpindle).ravel())/1e6
        hist, edge = np.histogram(diffTime, bins=np.arange(-30, 30, 1))
        hist = gaussian_filter1d(stats.zscore(hist), sigma=2)
        
        if 'Steve' in sub_name:
            
            m.append(plt.subplot(1, 2, 1))
            
            if 'Sleep' in sub_name:
            
                plt.plot(edge[0:len(hist)], hist, color='#f49e42', alpha=0.95-int(sub_name[-1])/10, label = 'Light'+ sub_name[-1])
                           
            else:

                plt.plot(edge[0:len(hist)], hist, color = '#726f6b', alpha= 0.95- int(sub_name[-1])/10)
            
            plt.title('Steve')
                
                
            
        if 'Ted' in sub_name:
            
            m.append(plt.subplot(1,2,2))
            if 'Sleep' in sub_name:
            
                plt.plot(edge[0:len(hist)], hist, color=colmapLight[int(sub_name[-1])], alpha= 1, label = 'Light'+ sub_name[-1])
                
            
            else:
                
                plt.plot(edge[0:len(hist)], hist, color = colmapDark[int(sub_name[-1])], alpha= 1 , label = 'Dark'+ sub_name[-1])
            
            plt.title('Ted')
                
        plt.legend(fontsize= 'xx-small', loc = 'best')

for pltind in [0,1]:
    m[pltind].set(xlabel='Time relative to PFC spindle (s)')

for pltind in [0]:
    m[pltind].set(ylabel='HPC spindle events (zscored)')      

plt.suptitle('PFC-Spindle vs HPC-Spindle') 

#plt.savefig(figFilename, dpi = 300)
    
    