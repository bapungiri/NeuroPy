import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stat

fileName = "/data/Clustering/SleepDeprivation/RatJ/Day1/RatJ_2019-05-31_03-55-36.eeg"


nChans = 8
SampFreq = 1250


Data = np.memmap(fileName, dtype="int16", mode="r")

Data1 = np.memmap.reshape(Data, (int(len(Data) / 75), 75))

chanData = Data1[:, 17]

zsc = np.abs(stat.zscore(chanData))

artifact_binary = np.where(zsc > 3, 0, 1)
artifact_binary = np.concatenate(([0], artifact_binary, [0]))
artifact_diff = np.diff(artifact_binary)


artifact_start = np.where(artifact_diff == 1)[0] / 1250
artifact_end = np.where(artifact_diff == -1)[0] / 1250


DatFileOG = "/data/Clustering/SleepDeprivation/RatJ/Day1/Shank2/RatJDay1_Shank2.dat"
DestFolder = "/data/Clustering/SleepDeprivation/RatJ/Day1/Shank2/Shank2DenoisedData.dat"


nChans = 8
SampFreq = 30000

b = []
for i in range(len(artifact_start)):

    start_time = artifact_start[i]
    end_time = artifact_end[i]

    duration = end_time - start_time  # in seconds
    b.append(
        np.memmap(
            DatFileOG,
            dtype="int16",
            mode="r",
            offset=2 * nChans * int(SampFreq * start_time),
            shape=(nChans * int(SampFreq * duration)),
        )
    )


c = np.memmap(DestFolder, dtype="int16", mode="w+", shape=sum([len(x) for x in b]))

del c
d = np.memmap(DestFolder, dtype="int16", mode="r+", shape=sum([len(x) for x in b]))

sizeb = [0]
sizeb.extend([len(x) for x in b])
sizeb = np.cumsum(sizeb)

for i in range(len(b)):

    d[sizeb[i] : sizeb[i + 1]] = b[i]
    # d[len(b[i]) : len(b1) + len(b2)] = b2
