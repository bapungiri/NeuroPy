#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 17:13:29 2019

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

with open("RatI_Day2Training.csv","a") as f:
    try:

        while True:
        #=== first lick trigger ==========
            if flag == 0 and GPIO.input(sensor1):
                GPIO.output(pump1,GPIO.HIGH)
                time.sleep(0.100)
                GPIO.output(pump1,GPIO.LOW)
                flag=2
                pump1_trig+=1

                writer = csv.writer(f,delimiter=",")
                writer.writerow([time.time(),2])

            if flag == 0 and GPIO.input(sensor2):
                GPIO.output(pump2,GPIO.HIGH)
                time.sleep(0.100)
                GPIO.output(pump2,GPIO.LOW)
                flag=1
                pump2_trig+=1

                writer = csv.writer(f,delimiter=",")
                writer.writerow([time.time(),2])


        #===== Alternate between wells ===========

            if flag == 0 and GPIO.input(sensor1):
                GPIO.output(pump1,GPIO.HIGH)
                time.sleep(0.100)
                GPIO.output(pump1,GPIO.LOW)
                flag=2
                pump1_trig+=1

                writer = csv.writer(f,delimiter=",")
                writer.writerow([time.time(),2])

            if flag == 0 and GPIO.input(sensor2):
                GPIO.output(pump2,GPIO.HIGH)
                time.sleep(0.100)
                GPIO.output(pump2,GPIO.LOW)
                flag=1
                pump2_trig+=1


            if flag == 0 and GPIO.input(sensor1):
                GPIO.output(pump1,GPIO.HIGH)
                time.sleep(0.100)
                GPIO.output(pump1,GPIO.LOW)
                flag=2
                pump1_trig+=1

                writer = csv.writer(f,delimiter=",")
                writer.writerow([time.time(),2])

            if flag == 1 and GPIO.input(sensor1):
                GPIO.output(pump1,GPIO.HIGH)
                time.sleep(0.100)
                GPIO.output(pump1,GPIO.LOW)
                flag=2
                pump1_trig+=1

            if flag == 2 and GPIO.input(sensor2):
                GPIO.output(pump2,GPIO.HIGH)
                time.sleep(0.100)
                GPIO.output(pump2,GPIO.LOW)
                flag=1
                pump2_trig+=1



            time.sleep(0.100) #necessary ottherwise cpu usage is 100%
    finally:
        GPIO.cleanup()
        print('\n','pump1=',pump1_trig,', pump2=', pump2_trig)

