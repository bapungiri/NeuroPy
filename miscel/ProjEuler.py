# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 22:15:21 2018

@author: Bapun
"""

#P251

#a =5
#b=1
#c=2
#
#CT = ((a+(c**0.5))**(1/3)+(a-(c**0.5))**(1/3) )
#
#print(CT)

#P291

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from pandas import DataFrame


iris = datasets.load_iris()
feat= iris.data
labels = iris.target
plt.plot(feat[labels==0,0],feat[labels==0,2],'r.')
plt.plot(feat[labels==1,0],feat[labels==2,2],'g.')
plt.plot(feat[labels==2,0],feat[labels==2,2],'k.')