# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 22:06:17 2018

@author: Bapun
"""
import pandas as pd
import statsmodels.formula.api as sm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
#import matplotlib.pyplot as plt
#plt.plot([1,2,3,4])
#plt.ylabel('some numbers')
#plt.show()

#colors = ['red', 'blue', 'green']
#print "colors[0]"

filename= '../../DataGen/Advertising.csv'

b = pd.read_csv(filename)
x1= np.array(b.TV)
x2= np.array(b.radio)
x3=np.array(b.newspaper)
y= np.array(b.sales)
results1 = sm.glm('sales ~ TV + radio+ newspaper', data=b).fit() 

xx, yy = np.meshgrid(range(10), range(10))
point  = np.array([1, 2, 3])
normal = np.array([1, 1, 2])
point2 = np.array([10, 50, 50])

# a plane is a*x+b*y+c*z+d=0
# [a,b,c] is the normal. Thus, we have to calculate
# d and we're set
d = -point.dot(normal)
# calculate corresponding z
z = (-normal[0] * xx - normal[1] * yy - d) * 1. /normal[2]

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot_surface(xx, yy, z, alpha=0.2)

# Ensure that the next plot doesn't overwrite the first plot
ax = plt.gca()
ax.hold(True)

ax.scatter(x1, x2,y)

#Axes3D.plot(x1, x2, y)