#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 16:29:41 2019

@author: bapung
"""

import numpy as np
import matplotlib.pyplot as plt
import os, fnmatch
from pathlib import Path
import pandas as pd
import scipy.stats as stat
from OsCheck import DataDirPath, figDirPath 

data_folder = Path(DataDirPath())
fig_name = figDirPath()+ 'MultiMazeFigures/' + 'CombinedSessions.pdf'

sourceDir = data_folder / 'MultiMazeData/'
fileDir = os.listdir(sourceDir)

pattern1 = 'sess*.npy'
SessNames = [] 
for entry in fileDir:  
    if fnmatch.fnmatch(entry, pattern1):
            SessNames.append(entry)
SessNames= np.sort(SessNames)

          
colmap = plt.cm.tab10(np.linspace(0,1,6))
numArms = [3,3,5,5,5,7]



sess = {}
t_track, x_track, z_track, subjects, runLogic = [], [], [],[], []
for session in [0,1,2,3,4,5]:
    
    sess_name = SessNames[session]
    chanceLevel = 1/numArms[session]
    
#    Allbehav = pd.read_csv(sourceDir / sess_name)
    Allbehav = np.load(sourceDir / sess_name)
    
#    Allbehav = pd.read_csv(sourceDir / sess_name)
#    AllCheck = pd.DataFrame.to_dict(Allbehav)
    
    AllCheck = Allbehav.item()
    
    sess[sess_name[0:8]] = AllCheck
    

    
#    subjects = allbehav.item().get('subjects')
    

np.save(data_folder / 'MultiMazeData/sessionAll.npy', sess)        
        
        
        
        