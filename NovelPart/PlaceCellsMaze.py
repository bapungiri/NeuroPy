# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 21:32:14 2019

@author: Bapun
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 10:00:45 2019

@author: bapung
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from OsCheck import DataDirPath
#import scipy.signal as sg
#import scipy.stats as stats
#from scipy.signal import hilbert
import h5py


sourceDir = DataDirPath() + 'wake_new/'

arrays = {}
f= h5py.File(sourceDir + 'wake-basics.mat', 'r') 
for k, v in f.items():
    arrays[k] = np.array(v)

#spks = {}
fspikes= h5py.File(sourceDir + 'testVersion.mat', 'r') 
fbehav= h5py.File(sourceDir + 'wake-behavior.mat', 'r') 
fpos= h5py.File(sourceDir + 'wake-position.mat', 'r') 
fspeed= h5py.File(sourceDir + 'wake-speed.mat', 'r') 
#fICAStrength = h5py.File('/data/DataGen/ICAStrengthpy.mat', 'r') 

subjects = arrays['basics']
#spikes = spks['spikes']

for sub in range(5,6):
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
    
    
    pyrid = [i for i in range(0,nUnits) if quality[i] < 4 and stability[i] == 1]
    cellpyr= [celltype[a] for a in pyrid]
    
    behav = np.transpose(fbehav['behavior'][sub_name]['time'][:])
    states = np.transpose(fbehav['behavior'][sub_name]['list'][:])
    posx = (fpos['position'][sub_name]['x'][:])
    posy = (fpos['position'][sub_name]['y'][:])
    post = (fpos['position'][sub_name]['t'][:])
    
    speed = (fspeed['speed'][sub_name]['v'][:]).squeeze()
    spdt = (fspeed['speed'][sub_name]['t'][:]).squeeze()
    
    
    
    posx_mz = posx[np.where((post > behav[1,0]) & (post < behav[1,1]))] 
    posy_mz = posy[np.where((post > behav[1,0]) & (post < behav[1,1]))] 
    post_mz = post[np.where((post > behav[1,0]) & (post < behav[1,1]))]
    
    xcoord = np.arange(min(posx_mz),max(posx_mz)+1,2)
    ycoord = np.arange(min(posy_mz),max(posy_mz)+1,2)
    xx,yy = np.meshgrid(xcoord,ycoord)
    
    plt.clf()
    for cell in range(len(pyrid)):
        spkt = cellpyr[cell].squeeze()
        spd_spk = np.interp(spkt,spdt,speed)
        
        spkt = spkt[spkt > 20]
        
        spktx = np.interp(spkt,post_mz,posx_mz)
        spkty = np.interp(spkt,post_mz,posy_mz)
        
        
            
        pf, xe, ye = np.histogram2d(spktx,spkty,bins = [xcoord,ycoord])
        pft = pf*(1/30)
        
        eps = np.spacing(1)
        pfRate = pf/(pft+eps)
        
        pfRate_smooth= gaussian_filter(pfRate, sigma=2)

#    ICA_strength = np.array(np.transpose(fICAStrength['ActStrength']['subjects'][sub_name]['wake'][:]))

    
        
    #    plt.plot(posx_mz,posy_mz,'.')
        plt.subplot(5,8,cell+1)
        plt.imshow(pfRate_smooth)
        
        plt.xticks([])
        plt.yticks([])
#        plt.subplot(1,2,2)
#        plt.imshow(pfRate_smooth)
    plt.suptitle(sub_name)
    