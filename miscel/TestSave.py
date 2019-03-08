# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 16:46:19 2019

@author: Bapun
"""

import numpy as np
import pandas as pd



t = np.linspace(0,1,1000000)
x = np.sin(t)

tel = {'jack': 4098, 'sape': 4139}

d = {'sub1': [1, t], 'sub2': [3, 4]}
df = pd.DataFrame(data=d, index=['t','x'])

df1 = df

df.to_hdf('foo.h5',key=['df', 'df1'],mode='w')


m1 = pd.read_hdf('foo.h5','df')

e = {'sub3': df}

df2 = pd.DataFrame

#np.savez('Test.npz', postrack = y)

#a = np.load('Test.npz')
