#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 11:36:40 2019

@author: bapung
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import fnmatch
from pathlib import Path
# import pandas as pd
import scipy.stats as stat
from OsCheck import DataDirPath, figDirPath

data_folder = Path(DataDirPath())
fig_name = figDirPath() + 'MultiMazeFigures/' + 'CombinedSessions.pdf'

sourceDir = data_folder / 'MultiMazeData/'
fileDir = os.listdir(sourceDir)

pattern1 = 'sess*.npy'
SessNames = []
for entry in fileDir:
    if fnmatch.fnmatch(entry, pattern1):
        SessNames.append(entry)
SessNames = np.sort(SessNames)


colmap = plt.cm.tab10(np.linspace(0, 1, 6))
numArms = [3, 3, 5, 5, 5, 7]

plt.clf()
t_track, x_track, z_track, subjects, runLogic = [], [], [], [], []
for session in [0, 1, 2, 3, 4, 5]:

    sess_name = SessNames[session]
    chanceLevel = 1/numArms[session]

#    Allbehav = pd.read_csv(sourceDir / sess_name)
    allbehav = np.load(sourceDir / sess_name)

    subjects = allbehav.item().get('subjects')
    num_sub = len(subjects)

    for sub in range(0, num_sub):
        sub_name = subjects[sub]
        runlogic = allbehav.item().get('runLogic')[sub]
        mov_sum_reward = []
        wind_size = 10
        if len(runlogic) > 10:
            for wind in range(0, len(runlogic)-wind_size+1):

                mov_sum_reward.append(sum(runlogic[wind:wind+wind_size]))
                run_avg_t = np.arange(wind_size, len(runlogic)+1)

        if len(runlogic) <= 10:

            mov_sum_reward.append(sum(runlogic))
            run_avg_t = len(runlogic)

        if len(mov_sum_reward) > 40:
            mov_sum_reward = mov_sum_reward[0:40]
            run_avg_t = run_avg_t[0:40]

        rand_val = np.random.random()*(4/100)-0.02

        binTest = [stat.binom_test(x, n=10, p=0.2, alternative='greater') for x in mov_sum_reward]

        binTest = np.where(np.asarray(binTest) < 0.05, 1, 0)
        percent_correct = [x / 10 for x in mov_sum_reward]
        percent_correct = [x+rand_val for x in percent_correct]

        plt.subplot(3, 2, session+1)
        if len(percent_correct) > 1:

            plt.plot(run_avg_t, percent_correct, label=sub_name,
                     color=colmap[sub], linewidth=1.5, alpha=0.95-sub/10, linestyle='-')

        else:

            plt.scatter(run_avg_t, percent_correct, label=sub_name,
                        color=colmap[sub], alpha=0.95-sub/10)


# ''        plt.plot(run_avg_t,percent_correct,label=sub_name,color= colmap[sub],
# linewidth = 2, alpha=0.95-sub/10, linestyle= '-')
        plt.ylim(0, 1.1)
        plt.ylabel('Proportion correct')
        plt.xlabel('# Choices')
        plt.title('Session' + str(session+1), loc='left')
    plt.plot([0, 50], [chanceLevel, chanceLevel], label='Chance',
             color='#a1a7af', linestyle='--', linewidth=1.5)


plt.legend(ncol=3)
plt.suptitle('MultiArmMaze behavior')
plt.savefig(fig_name, dpi=150)
