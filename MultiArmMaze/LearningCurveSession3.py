#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 15:23:42 2019

@author: bapung
"""


import datetime
import time
import os, fnmatch
from pathlib import Path
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from OsCheck import DataDirPath, figDirPath 
import scipy.signal as sg
import scipy.ndimage as filt 



data_folder = Path(DataDirPath())

sourceDir = data_folder / 'MultiMazeData/session3/'
fileDir = os.listdir(sourceDir)
pattern1 = '*Take*'
pattern2 = '*Sess3.csv'  
filePosNames = [] 
SensorNames = [] 
for entry in fileDir:  
    if fnmatch.fnmatch(entry, pattern1):
            filePosNames.append(entry)
    if fnmatch.fnmatch(entry, pattern2):
            SensorNames.append(entry)


plt.clf()
for sub in [0,2,3,4,5]:

    PosFile = filePosNames[sub]
    sub_name= PosFile[0:4]
    print(sub_name)
#    tbegin = datetime.datetime(2019, 2, 18, 17, 23, 21,0)
#    datetime_object = datetime.datetime.strptime(PosFile[20:-7], '%Y-%m-%d %I.%M.%S')
    tbegin = datetime.datetime.strptime(PosFile[20:-7]+'.0', '%Y-%m-%d %H.%M.%S.%f')
    tbegin = time.mktime(tbegin.timetuple()) + tbegin.microsecond / 1E6
    
    file1 = [] 
    for entry in SensorNames:  
        if fnmatch.fnmatch(entry, PosFile[0:4]+'*'):
                file1 = sourceDir / entry
    file2 = sourceDir / PosFile
    
    mazeCoord = [72,112, 168, 283, 325]

    
    opti = pd.read_csv(file2,skiprows=range(0, 6))
    numColData = opti.columns.tolist() 
    t = opti['Time (Seconds)']
    
    x=opti.X
    y=opti.Y
    z=opti.Z
    for i in range(1,int((len(numColData)-2)/3)):
        
        if i < int((len(numColData)-2)/3)-1:
            x1 = opti['X.'+str(i)]
            x2 = opti['X.'+str(i+1)]
            valx1 = pd.Series.first_valid_index(x1)
            valx2 = pd.Series.first_valid_index(x2)        
            x = pd.concat([x[0:valx1],x1[valx1:valx2]])
            
            z1 = opti['Z.'+str(i)]
            z2 = opti['Z.'+str(i+1)]
            valz1 = pd.Series.first_valid_index(z1)
            valz2 = pd.Series.first_valid_index(z2)        
            z = pd.concat([z[0:valz1],z1[valz1:valz2]])
        
        if i == int((len(numColData)-2)/3)-1:
            x1 = opti['X.'+str(i)]
            valx1 = pd.Series.first_valid_index(x1)
            x = pd.concat([x[0:valx1],x1[valx1:len(x1)+1]])
            
            z1 = opti['Z.'+str(i)]
            valz1 = pd.Series.first_valid_index(z1)
            z = pd.concat([z[0:valx1],z1[valx1:len(z1)+1]])
            
#        y1 = opti['Y.'+str(i)]
#        valy = pd.Series.first_valid_index(y1)        
#        y = pd.concat([y[0:valy],y1])
#        
#        z1 = opti['Z.'+str(i)]
#        valz = pd.Series.first_valid_index(z1)
#        z = pd.concat([z[0:valz],z1])

    
    xedges = np.linspace(min(x),max(x),200)
    yedges = np.linspace(min(z),max(z),200)
    
    
    
    
    
    df = pd.read_csv(file1,header=None)
    time1 = df[0] #you can also use df['column_name']
    time1 = time1[~time1.isnull()]
    
    Sensor1 = df[1] #you can also use df['column_name']
    Sensor1 = Sensor1[~Sensor1.isnull()]
    
    time_n = time1-tbegin
    
    
    if time_n[0]>40000:
        time_n= time_n-12*60*60
    
    
    sensorX = np.interp(time_n,t,x)
    sensorZ = np.interp(time_n,t,z)
    
    
    radi = np.sqrt((x**2+z**2))
    vel = np.diff(radi)
    
    ratioCoord = z/x
    
    Thresh = np.where(radi > 0.5)
    angle = np.arctan2(z,x)
    
    hist_angle, angles = np.histogram(angle[~np.isnan(angle)],np.linspace(-np.pi,np.pi,40))
    
    arm_angle_id, arm_info = sg.find_peaks(hist_angle,threshold=2500)
    arm_angle = (angles[arm_angle_id])*(180/np.pi)
    
    
    
    
    
    radi_thresh = radi-0.4
    
    
    angle_cross = []
    time_cross=[]
    outward= 0
    for i in range(0,len(radi_thresh)):
        if (radi_thresh[i]<0 and radi_thresh[i+1]>0):
            outward  = outward+1
            angle_cross.append(angle[i])
            time_cross.append(t[i])
            
    
    angle_cross = np.asarray(angle_cross)
    time_cross = np.asarray(time_cross)
    
    angle_degree  = (angle_cross/np.pi)*180
    
    run_logic = np.where([(angle_degree > 90) & (angle_degree <180)],1,0).squeeze()
    time_false = time_cross[run_logic==0]
    time_true = time_cross[run_logic==1]
    
    reward_arm_angle = angle_degree[run_logic==1]
    
    reward_logic = np.where(np.abs(np.diff(reward_arm_angle))>30,1,0)
    
    
    k = np.where(reward_logic==1)[0]
    k = k+1
    k = np.append([0],k)
    
    
    
    time_corr_choice =time_true[k]
    time_incorr_choice = time_true[np.where(reward_logic==0)]
    
    
    ##==== Running Avergae Calculation ===============
    
    run_avg = np.zeros(len(angle_degree))
    for ang1 in range(0,len(angle_degree)):
        
        deg = angle_degree[ang1]
        if (sum(run_avg)==0):
            
            if abs(deg-110)<10 or abs(deg-162)<10:
                run_avg[ang1] = 1
                temp = deg
                
        if (sum(run_avg)!=0):
            
            if abs(deg-110)<10 and abs(temp-deg)>10:
                run_avg[ang1] = 1
                temp=deg
                
        if abs(deg-162)<10 and abs(temp-deg)>10:
                run_avg[ang1] = 1
                temp=deg
     
        
    mov_sum_reward=[]
    wind_size = 10
    for wind in range(0,len(run_avg)-wind_size+1):
         
         mov_sum_reward.append(sum(run_avg[wind:wind+wind_size]))
         
         
    run_avg_t = np.arange(wind_size,len(run_avg)+1)     
         
     
        
        
        
#        
#        if (ang1-110)<10:
#            
#            run_avg[ang1] = 1
#            
#        if (ang1-162)<10:
#            
#            run_avg[ang1] = -1
            
            
    
    
    
    #for i in range(0,len(reward_arm_angle)-1):
    #    if (reward_arm_angle[i]*reward_arm_angle[i+1]<0):
    #        corr_choice  = corr_choice+1
    #        time_corr_choice.append(time_true[i])
    #        
    #    if (reward_arm_angle[i]*reward_arm_angle[i+1]>0):
    #        corr_choice  = corr_choice+1
    #        time_corr_choice.append(time_true[i])
    #
    #time_corr_choice = np.asarray(time_corr_choice)
     
    
    timeRecord = t[pd.Series.last_valid_index(x)]
    
    over_time_false, edges = np.histogram(np.concatenate((time_false,time_incorr_choice),axis=0),np.linspace(0,timeRecord,10))
    over_time_true, edges = np.histogram(time_corr_choice,np.linspace(0,timeRecord,10))
    
    false_choice_t = np.cumsum(over_time_false)
    correct_choice_t = np.cumsum(over_time_true)
    
    ti = (edges[0:-1]+5*60)/60
    
    reward_sensor = np.where((Sensor1==4) | (Sensor1==6))[0]
    time_sensor_reward = time_n[reward_sensor[0::]]
    true_sensor, edges = np.histogram(time_sensor_reward,np.linspace(0,timeRecord,10))
    true_sensor_t = np.cumsum(true_sensor)
    
    
    plt.subplot(6,2,sub*2+1)
#    plt.plot(ti,false_choice_t,'r', label = 'Wrong Arm')
#    plt.plot(ti, correct_choice_t,'g',label = 'Correct Arm')
#    plt.plot(ti, true_sensor_t,'m',label = 'Sensor reward')
    plt.plot(run_avg_t,mov_sum_reward,label='Running avg.('+str(wind_size)+'runs)')
    
    plt.ylabel('# Correct choices')
    
    if sub==0:
        plt.legend()
        plt.xlabel('trial')
    plt.title(sub_name)
    
    
    time_epoch = int(timeRecord/4)*120
    
    
    for i in [3]:
        
        xt = x[i:time_epoch*(i+1)]
        zt = z[i:time_epoch*(i+1)]
        
        hist_pos= np.histogram2d(xt[~np.isnan(xt)],zt[~np.isnan(zt)],bins = (xedges,yedges)) 
        
        dsf = filt.gaussian_filter(hist_pos[0],sigma=1.5, order=0)
        
        plt.subplot(6,2,sub*2+2)
        plt.imshow(dsf,cmap='viridis',vmin = 0,vmax=200)
    
    
    
    
#    fig = plt.figure(1)
    ##plt.plot(hist_angle[0])
#    ax = fig.add_subplot(121, projection='polar')
#    c = ax.scatter(angle, radi)
    ##ax = fig.add_subplot(122)
    ##c = ax.plot(radi-0.4)
    
    
    
    
    #a = [time.localtime(x) for x in time1.values]
    
    numData = int(np.floor(len(x)/4))
    
    #plt.clf()
    #
    #for i in range(0,4):
    #    plt.subplot(1,4,i+1)
    #    #plt.plot(time1,Sensor1,'.')
    #    #plt.subplot(1,2,2)
    #    plt.plot(x[i*numData:(i+1)*numData],z[i*numData:(i+1)*numData])
    #    #plt.plot(sensorX,sensorZ,'r.')
    ##    plt.title('RatC Session 1')
    
    #
    
    #with open(file1) as csvfile:
    #    readCSV = csv.reader(csvfile, delimiter=',')
    #    dates = []
    #    colors = []
    #    for row in readCSV:
    #        color = row[1]
    #        date = row[0]
    #
    #        dates.append(date)
    #        colors.append(color)