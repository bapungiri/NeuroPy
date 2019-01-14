# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 21:40:10 2019

@author: Bapun
"""

import numpy as np
import matplotlib.pyplot as plt	
import time as tm

dt = 0.01
time = np.arange(0,500,dt)


reward_x = 10*np.random.random()
reward_y = 10*np.random.random()

x = 10*np.random.rand(25,)
y = 10*np.random.rand(25,)

scale = np.ones(25,)
for t in time:
    
    dist = np.sqrt((x-reward_x)**2 + (y-reward_y)**2)
    win = np.where(dist < 0.2)
    scale[win] = scale[win]+2
    
    offlimx = np.where(x >10)
    offlimy = np.where(y >10)
    offlimxneg = np.where(x <0)
    offlimyneg = np.where(y <0)
    
    if offlimx or offlimy or offlimxneg or offlimyneg:
        
        x[offlimx] = 10-0.5
        y[offlimy] = 10-0.5
        x[offlimxneg] = 0+0.5
        y[offlimyneg] = 0+0.5

#    print(np.where(x >10))
#    print(np.where(y >10))
        
    
    if win:
        reward_x = 10*np.random.random()
        reward_y = 10*np.random.random()
    
    
    theta = 2*np.pi*np.random.rand(25,)
    x = x+np.cos(theta)
    y = y+np.sin(theta)
    
    
    
    
plt.clf()
plt.plot(reward_x,reward_y,'k.')
plt.scatter(x,y,scale)
plt.xlim(0,10)
plt.ylim(0,10)
#    plt.pause(0.0001)
    


