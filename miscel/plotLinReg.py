# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 15:29:36 2018

@author: Bapun
"""


import pandas as pd
import statsmodels.formula.api as sm
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as polyfit
#import sys
#import matplotlib.pyplot as plt
#plt.plot([1,2,3,4])
#plt.ylabel('some numbers')
#plt.show()

#colors = ['red', 'blue', 'green']
#print "colors[0]"

filename= '../../DataGen/Auto.csv'

mpg_data = pd.read_csv(filename)

y= np.array(mpg_data.mpg)
hp = np.array(mpg_data.displacement)
#results1 = sm.glm('mpg ~ horsepower', data=mpg_data).fit()

lin_mod = polyfit.linregress(hp,y)

yhat = lin_mod[0]*hp+lin_mod[1]
y_res = y-yhat


plt.scatter(hp,y_res)