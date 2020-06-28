#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter

from callfunc import processData

#%% Subjects
basePath = [
    "/data/Clustering/SleepDeprivation/RatJ/Day1/",
    "/data/Clustering/SleepDeprivation/RatK/Day1/",
    "/data/Clustering/SleepDeprivation/RatN/Day1/",
    # "/data/Clustering/SleepDeprivation/RatJ/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day2/",
    # "/data/Clustering/SleepDeprivation/RatN/Day2/",
]


sessions = [processData(_) for _ in basePath]


#%% Stabiliy of cells using firing rate
#region
plt.clf()
fig = plt.figure(1, figsize=(10, 15))
gs = gridspec.GridSpec(3, 1, figure=fig)
fig.subplots_adjust(hspace=0.3)
fig.suptitle("Change in interspike interval during Sleep Deprivaton")

for sub, sess in enumerate(sessions):

    sess.trange = np.array([])
    spikes = sess.spikes.times
    pre = sess.epochs.pre
    post = sess.epochs.post
    spkinfo = sess.spikes.info
    reqcells_id = np.where(spkinfo["q"] < 4)[0]
    # spikes = [spikes[cell] for cell in reqcells_id]

    first_hour = pre[]

    frate_pre = [np.histogram(cell, bins=pre)[0] / np.diff(pre) for cell in spikes]
    frate_post = [np.histogram(cell, bins=post)[0] / np.diff(post) for cell in spikes]

    ax = fig.add_subplot(gs[sub])
    ax.plot(frate_pre, frate_post, ".")
    ax.plot([0, max(frate_pre)], [0, max(frate_pre)])
    # ax.axis("equal")
#endregion


#%% Stabiliy of cells during Sleep deprivation
#region
plt.clf()
fig = plt.figure(1, figsize=(10, 15))
gs = gridspec.GridSpec(3, 2, figure=fig)
fig.subplots_adjust(hspace=0.3)
fig.suptitle("Change in interspike interval during Sleep Deprivaton")

for sub, sess in enumerate(sessions):

    sess.trange = np.array([])
    spikes = sess.spikes.times
    pre = sess.epochs.pre
    post = sess.epochs.post
    spkinfo = sess.spikes.info

    # reqcells_id = np.where(spkinfo["q"] < 8)[0]
    # spikes = [spikes[cell] for cell in reqcells_id]

    mua = np.concatenate(spikes)

    first_hour = [post[0],post[0]+3600]
    last_hour = [post[0]+4*3600,post[0]+5*3600]
    sd_period= np.linspace(post[0],post[0]+5*3600,6)

    frate_1h = [np.histogram(cell, bins=first_hour)[0] / np.diff(pre) for cell in spikes]
    frate_5h = [np.histogram(cell, bins=last_hour)[0] / np.diff(post) for cell in spikes]

    mua_frate_sd = np.histogram(mua,bins=sd_period)[0]/np.diff(sd_period)

    subname = sess.sessinfo.session.sessionName
    ax = fig.add_subplot(gs[sub,0])
    ax.plot(np.log10(frate_1h), np.log10(frate_5h), ".",color='#616161')
    ax.plot([-4,1],[-4,1],'r')
    ax.set_xlabel('Firing rate (1st hour SD)')
    ax.set_ylabel('Firing rate (5th hour SD)')
    ax.set_title(subname,loc='left')

    ax = fig.add_subplot(gs[sub,1])
    ax.bar(np.arange(5),mua_frate_sd,color='#9E9E9E')
    ax.set_xticks([0,1,2,3,4])
    ax.set_xticklabels(['ZT'+str(i+1) for i in range(5)])
    ax.set_ylabel('MUA firing rate (Hz)')
    # ax.plot([0, max(frate_pre)], [0, max(frate_pre)])
    # ax.axis("equal")
#endregion

