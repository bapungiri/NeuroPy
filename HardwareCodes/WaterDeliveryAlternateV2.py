#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 11:16:06 2019

@author: bapung
"""

import RPi.GPIO as GPIO
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
GPIO.setup(pump1,GPIO.OUT, initial =0)

GPIO.setup(sensor1,GPIO.IN)
GPIO.setup(sensor2,GPIO.IN)

#=====Initial values ============
flag = 0
pump1_trig=0
pump2_trig=0


def pumpTrue(whichpump):
    GPIO.output(whichpump,GPIO.HIGH)
    time.sleep(0.100)
    GPIO.output(whichpump,GPIO.LOW)
    return flag



with open("RatI_Day2Training.csv","a") as f:
    try:

        while True:

        #===== Alternate between wells ===========

            if (flag == 0 or flag==1) and GPIO.input(sensor1):
                pumpTrue(pump1,flag)
                flag=2
                writer = csv.writer(f,delimiter=",")
                writer.writerow([time.time(),1])

            if (flag == 0 or flag==2) and GPIO.input(sensor2):
                pumpTrue(pump2,flag)
                flag=1
                writer = csv.writer(f,delimiter=",")
                writer.writerow([time.time(),3])


            if flag == 2 and GPIO.input(sensor1):
                writer = csv.writer(f,delimiter=",")
                writer.writerow([time.time(),2])

            if flag == 1 and GPIO.input(sensor2):
                writer = csv.writer(f,delimiter=",")
                writer.writerow([time.time(),4])


            time.sleep(0.100) #necessary ottherwise cpu usage is 100%
    finally:
        GPIO.cleanup()
        print('\n','pump1=',pump1_trig,', pump2=', pump2_trig)

