import numpy as np
import matplotlib.pyplot as plt


filename = "/data/Clustering/SleepDeprivation/RatN/Day2/RatN_Day2_2019-10-11_03-58-54_part1.dat"

Destfile = "/data/Clustering/SleepDeprivation/RatN/Day2/RatN_Day2_2019-10-11_03-58-54_smallperiod.dat"

nChans = 134
SampFreq = 30000
duration1 = 20  # from this time in seconds
duration2 = 60 * 60  # duration of chunk

# read required chunk from the source file
b1 = np.memmap(
    filename,
    dtype="int16",
    offset=2 * nChans * SampFreq * duration1,
    mode="r",
    shape=(nChans * SampFreq * duration2),
)

# allocates space for the file
c = np.memmap(Destfile, dtype="int16", mode="w+", shape=(len(b1)))
c[: len(b1)] = b1
# del c

# # writes the data to that space
# d = np.memmap(Destfile, dtype="int16", mode="r+", shape=(len(b1)))
