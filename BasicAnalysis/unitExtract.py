import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


workDir = '~/Documents/ClusteringHub/RatJ_2019-05-31_03-55-36/Shank7-8/'
workDir1 = '~/Documents/ClusteringHub/RatJ_2019-05-31_03-55-36/'

dir_path = os.path.dirname(workDir1)
dir_in = os.walk(dir_path)

file = workDir+'Shank7-8.csv'

spkInfo = pd.read_csv(file, header=None, names=[
    'spktimes', 'ClusterNum', 'MaxSite'])

spktimes = spkInfo['spktimes']

histspk, edges = np.histogram(spktimes, 14)

plt.plot(histspk)
