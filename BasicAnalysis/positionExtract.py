import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


filename = '/data/Clustering/SleepDeprivation/RatJ/Behavior_Position/Take 2019-05-31 03.55.25 AM.fbx'

f = open(filename, 'r')
# data = pd.read_csv(filename, header=5)

# time = data['Time (Seconds)'].tolist()
# x = data['X'].tolist()
# y = data['Y'].tolist()
# z = data['Z'].tolist()


# plt.plot(x, z, '.')

i = 1
k = 830
xpos, ypos, zpos = [], [], []
with open(filename) as f:
    next(f)
    for line in f:
        i = i+1
        if i > k:
            line = line.strip()
            m = line.split(',')

            xpos.append(m[1])
            ypos.append(m[2])
            zpos.append(m[3])

        if i > (k+120*60*60*2):
            break

xpos = list(map(float, xpos))
ypos = list(map(float, ypos))
zpos = list(map(float, zpos))
plt.plot(zpos)
