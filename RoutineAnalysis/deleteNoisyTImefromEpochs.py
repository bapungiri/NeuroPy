import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from datetime import datetime


basePath = Path("/data/Clustering/SleepDeprivation/RatJ/Day1/")
for file in os.listdir(basePath):
    if file.endswith(".eeg"):

        subname = file[:-4]
        filename = os.path.join(basePath, file)
        filePrefix = os.path.join(basePath, file[:-4])

nframes = np.load(basePath / "og_files" / "numframes_OG.npy")
noisyFrames = np.load(basePath / "og_files" / "noisy_timestamps_fromOG.npy")
epochs = np.load(
    basePath / "RatJ_Day1_2019-05-31_03-55-36_epochs.npy", allow_pickle=True
)
posInfo = np.load(
    basePath / "RatJ_Day1_2019-05-31_03-55-36_position.npy", allow_pickle=True
)

posX = posInfo.item().get("X")
posY = posInfo.item().get("Y")
frames = posInfo.item().get("frames")
video_starttime = posInfo.item().get("begin")

ephys_starttime = datetime.strptime(subname[-19:], "%Y-%m-%d_%H-%M-%S")

time_diff = video_starttime - ephys_starttime

plt.plot(posY)

