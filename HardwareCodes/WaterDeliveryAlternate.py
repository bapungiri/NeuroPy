#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 17:13:29 2019

@author: bapung
"""

import RPi.GPIO as GPIO
import time
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)


#=======Setting up I/O===========

GPIO.setup(18,GPIO.OUT)
GPIO.setup(23,GPIO.OUT)

GPIO.setup(22,GPIO.IN, pull_up_down= GPIO.PUD_down)
GPIO.setup(7,GPIO.IN)

flag = 0

try:

    while True:

        if flag == 0 and GPIO.input(22):
            GPIO.output(18,GPIO.HIGH)
            time.sleep(0.100)
            GPIO.output(18,GPIO.LOW)
            flag==2

        if flag == 0 and GPIO.input(7):
            GPIO.output(18,GPIO.HIGH)
            time.sleep(0.100)
            GPIO.output(18,GPIO.LOW)
            flag==1




        if flag == 1 and GPIO.input(22):
            GPIO.output(18,GPIO.HIGH)
            time.sleep(0.100)
            GPIO.output(18,GPIO.LOW)
            flag==2

        if flag == 2 and GPIO.input(7):
            GPIO.output(23,GPIO.HIGH)
            time.sleep(0.100)
            GPIO.output(23,GPIO.LOW)
            flag==1

finally:
    GPIO.cleanup()
