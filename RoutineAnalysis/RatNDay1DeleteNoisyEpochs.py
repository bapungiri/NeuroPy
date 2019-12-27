import numpy as np

for i in range(2, 9):

    filename = (
        "/data/Clustering/SleepDeprivation/RatN/Day1/Shank"
        + str(i)
        + "/Shank"
        + str(i)
        + ".dat"
    )

    nChans = 16
    SampFreq = 30000
    duration = 191 * 60  # in seconds
    b1 = np.memmap(
        filename, dtype="int16", mode="r", shape=(nChans * SampFreq * duration)
    )

    duration2 = 192 * 60  # seconds
    b2 = np.memmap(
        filename, dtype="int16", mode="r", offset=2 * nChans * SampFreq * duration2
    )

    DestFile = (
        "/data/Clustering/SleepDeprivation/RatN/Day1/Shank"
        + str(i)
        + "/RatNDay1Shank"
        + str(i)
        + ".dat"
    )
    c = np.memmap(DestFile, dtype="int16", mode="w+", shape=(len(b1) + len(b2)))

    del c
    d = np.memmap(DestFile, dtype="int16", mode="r+", shape=(len(b1) + len(b2)))
    d[: len(b1)] = b1
    d[len(b1) : len(b1) + len(b2)] = b2

