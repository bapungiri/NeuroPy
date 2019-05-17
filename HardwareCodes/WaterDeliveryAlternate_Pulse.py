#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 10:30:27 2019

@author: bapung

Alternate between water wells with  pulses sent to open ephys via hdmi for syncing post processing in dat files

"""


import RPi.GPIO as GPIO
import datetime as dt
import time
import csv
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)


pump1= 16
pump2= 18
pumpOpen = 0.120 #seconds

sensor1=22
sensor2=7

#=======Setting up I/O===========

GPIO.setup(pump1,GPIO.OUT, initial=0)
GPIO.setup(pump2,GPIO.OUT, initial =0)

GPIO.setup(sensor1,GPIO.IN)
GPIO.setup(sensor2,GPIO.IN)

#=====Initial values ============
flag = 0
pump1_trig=0
pump2_trig=0


def pumpTrue(whichpump,sensorState):
    GPIO.output(whichpump,GPIO.HIGH)
    time.sleep(pumpOpen)
    GPIO.output(whichpump,GPIO.LOW)
    writer.writerow([time.time()-start_time,sensorState])
    return flag

def pumpFalse(sensorState):
    writer = csv.writer(f,delimiter=",")
    writer.writerow([time.time()-start_time,sensorState])

Date_now = dt.datetime.now().strftime("%Y-%m-%d_%H:%M")
filename = 'RatI_Day2Training_'+ Date_now + '.csv'


with open(filename,"a") as f:

    start_time = time.time()
    writer = csv.writer(f,delimiter=",")
    writer.writerow(['Start Time = ', start_time])
    writer.writerow(['Time','SensorState'])

    try:

        while True:

        #===== Alternate between wells ===========

            if (flag == 0 or flag==1) and GPIO.input(sensor1):
                sensorConfig=1
                pumpTrue(pump1,sensorConfig)
                flag=2
                pump1_trig+=1

            if flag == 2 and GPIO.input(sensor1):
                sensorConfig=2
                pumpFalse(sensorConfig)
                time.sleep(0.200)

            if (flag == 0 or flag==2) and GPIO.input(sensor2):
                sensorConfig=3
                pumpTrue(pump1,sensorConfig)
                flag=1
                pump2_trig+=1


            if flag == 1 and GPIO.input(sensor2):
                sensorConfig=4
                pumpFalse(sensorConfig)
                time.sleep(0.200)


            time.sleep(0.100) #necessary ottherwise cpu usage is 100%
    finally:
        GPIO.cleanup()
        print('pump1=',pump1_trig, ',pump2=', pump2_trig)

