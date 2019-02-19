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


df = pd.read_csv(file1)
time1 = df.time #you can also use df['column_name']
time1 = time1[~time1.isnull()]

Sensor1 = df.sensor #you can also use df['column_name']
Sensor1 = Sensor1[~Sensor1.isnull()]

time_n = time1-tbegin


sensorX = np.interp(time_n,t,x)
sensorZ = np.interp(time_n,t,z)





#a = [time.localtime(x) for x in time1.values]


plt.clf()
#plt.subplot(1,2,1)
#plt.plot(time1,Sensor1,'.')
#plt.subplot(1,2,2)
plt.plot(x,z,'.')
plt.plot(sensorX,sensorZ,'r.')

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