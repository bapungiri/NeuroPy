#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 16:50:07 2019

@author: bapung
"""


import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter,gaussian_filter1d 
from OsCheck import DataDirPath, figDirPath
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

figFilename = figDirPath() +'PlaceCells.pdf'
pdf = PdfPages(figFilename)

for sub in [6]:
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
    nPyr = len(pyrid)
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
    
    dist_maze = np.zeros(len(posx_mz))
    
    min_val = min(posx_mz)
    
    
    for d in range(0,len(posx_mz)):
    
        coord = [posx_mz[d],posy_mz[d]]
        
        if coord[1] < 100:
            dist_maze[d] = coord[0]
            
        if coord[1] > 300 and coord[0] < 530:
            dist_maze[d] = 570+388+530-coord[0]
            
        else:
            dist_maze[d]= 570+coord[1]-100
            
    dist_maze = dist_maze
    
    eps = np.spacing(1)
    lin_coord = np.linspace(0,2*(max(posx_mz)-min(posx_mz))+max(posy_mz)-min(posy_mz),500)
    lin_time, lin_bin = np.histogram(dist_maze,bins = lin_coord)
    lin_time = lin_time*(1/30)
    lin_time = lin_time+ eps
    
    
    xcoord = np.arange(min(posx_mz),max(posx_mz)+1,2)
    ycoord = np.arange(min(posy_mz),max(posy_mz)+1,2)
    xx,yy = np.meshgrid(xcoord,ycoord)
    
    
    
    plt.clf()
    for cell in range(len(pyrid)):
        spkt = cellpyr[cell].squeeze()
        spd_spk = np.interp(spkt,spdt,speed)
        
        spkt = spkt[spd_spk > 5]   #only selecting spikes where rat's speed is  > 5 cm/s
        
        spktx = np.interp(spkt,post_mz,posx_mz)
        spkty = np.interp(spkt,post_mz,posy_mz)
        
        dist = np.zeros(len(spktx))
        for d in range(0,len(spktx)):
            
            coord = [spktx[d],spkty[d]]
            
            if coord[1] < 100:
                dist[d] = coord[0]
            
            if coord[1] > 300 and coord[0] < 530:
                dist[d] = 570+388+530-coord[0]
                
            else:
                dist[d]= 570+coord[1]-100
        
            
        pf, xe, ye = np.histogram2d(spktx,spkty,bins = [xcoord,ycoord])
        dist = dist
        
        
        pf1, xe1= np.histogram(dist,bins = lin_coord)
        
        
        
        lin_pf = gaussian_filter1d(pf1,3)
        lin_pft = gaussian_filter1d(lin_time,3)
        
#        pfRate_smooth_lin = pf1/(lin_time)
        pfRate_smooth_lin = lin_pf/lin_pft
        
        
        
        
        pft = pf*(1/30)
        
        
        
        pfRate = pf/(pft+eps)
        
        pfRate_smooth= gaussian_filter(pfRate, sigma=3)
        
        
        
        

#    ICA_strength = np.array(np.transpose(fICAStrength['ActStrength']['subjects'][sub_name]['wake'][:]))
        
        nRows = np.ceil(np.sqrt(nPyr))
        nCols = np.ceil(np.sqrt(nPyr))
    
        
    #    plt.plot(posx_mz,posy_mz,'.')
        plt.subplot(nRows,nCols,cell+1)
#        plt.imshow(pfRate_smooth)
        plt.plot(xe1[0:-1],pfRate_smooth_lin)
        
#        plt.xticks([])
        plt.yticks([])
#        plt.subplot(1,2,2)
#        plt.imshow(pfRate_smooth)
    plt.suptitle(sub_name)
    plt.tight_layout()
#    pdf.savefig(dpi=300)

pdf.close()    
    