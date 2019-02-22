#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 15:43:54 2019

@author: bapung
"""
import datetime
import time
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from OsCheck import DataDirPath, figDirPath 


sourceDir = DataDirPath() + 'MultiMazeData/'

tbegin = datetime.datetime(2019, 2, 18, 17, 23, 21,0)
tbegin = time.mktime(tbegin.timetuple()) + tbegin.microsecond / 1E6

file1 = sourceDir+'RatC_Day1Sess1.csv'
file2 = sourceDir+'Take 2019-02-18 05.23.21 PM.csv'


opti = pd.read_csv(file2,skiprows=range(0, 6))
t = opti['Time (Seconds)']
x = opti.X
y = opti.Y
z = opti.Z

xedges = np.linspace(min(x),max(x),200)
yedges = np.linspace(min(z),max(z),200)


hist_pos = np.histogram2d(x[~np.isnan(x)],z[~np.isnan(z)],bins = (xedges,yedges)) 


df = pd.read_csv(file1)
time1 = df.time #you can also use df['column_name']
time1 = time1[~time1.isnull()]

Sensor1 = df.sensor #you can also use df['column_name']
Sensor1 = Sensor1[~Sensor1.isnull()]

time_n = time1-tbegin


sensorX = np.interp(time_n,t,x)
sensorZ = np.interp(time_n,t,z)


radi = np.sqrt((x**2+z**2))
vel = np.diff(radi)

ratioCoord = z/x

Thresh = np.where(radi > 0.5)
angle = np.arctan2(z,x)

hist_angle = np.histogram(angle[~np.isnan(angle)])


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

run_logic = np.where([(angle_degree < 70) & (angle_degree >20)],0,1).squeeze()
time_false = time_cross[run_logic==0]
time_true = time_cross[run_logic==1]

reward_arm_angle = angle_degree[run_logic==1]

reward_logic = np.where(np.abs(np.diff(reward_arm_angle))>20,1,0)

 


time_corr_choice =time_true[np.where(reward_logic==1)]
time_incorr_choice = time_true[np.where(reward_logic==0)]

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
 



over_time_false, edges = np.histogram([time_false,time_incorr_choice],np.linspace(0,2800,10))
over_time_true, edges = np.histogram(time_corr_choice,np.linspace(0,2800,10))

false_choice_t = np.cumsum(over_time_false)
correct_choice_t = np.cumsum(over_time_true)

ti = (edges[0:-1]+5*60)/60

plt.clf()

plt.plot(ti,false_choice_t,'r', label = 'Wrong Arm')
plt.plot(ti, correct_choice_t,'g',label = 'Correct Arm')
plt.xlabel('time')
plt.ylabel('Cummulative choices')
plt.legend()

#fig = plt.figure(1)
##plt.plot(hist_angle[0])
#ax = fig.add_subplot(121, projection='polar')
#c = ax.scatter(angle, radi)
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