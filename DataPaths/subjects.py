from callfunc import processData


def allsess():
    """all data folders together """
    paths = [
        "/data/Clustering/SleepDeprivation/RatJ/Day1/",
        "/data/Clustering/SleepDeprivation/RatK/Day1/",
        "/data/Clustering/SleepDeprivation/RatN/Day1/",
        "/data/Clustering/SleepDeprivation/RatJ/Day2/",
        "/data/Clustering/SleepDeprivation/RatK/Day2/",
        "/data/Clustering/SleepDeprivation/RatN/Day2/",
        "/data/Clustering/SleepDeprivation/RatJ/Day3/",
        "/data/Clustering/SleepDeprivation/RatK/Day3/",
        "/data/Clustering/SleepDeprivation/RatN/Day3/",
        "/data/Clustering/SleepDeprivation/RatJ/Day4/",
        "/data/Clustering/SleepDeprivation/RatK/Day4/",
        "/data/Clustering/SleepDeprivation/RatN/Day4/",
        "/data/Clustering/SleepDeprivation/RatA14d1LP/Rollipram/",
    ]
    return [processData(_) for _ in paths]


def sd(indx=None):
    """Sleep deprivation sessions"""

    paths = [
        "/data/Clustering/SleepDeprivation/RatJ/Day1/",
        "/data/Clustering/SleepDeprivation/RatK/Day1/",
        "/data/Clustering/SleepDeprivation/RatN/Day1/",
    ]
    if indx is not None:
        paths = [paths[_] for _ in indx]
    return [processData(_) for _ in paths]


def nsd(indx=None):
    """Control sessions for sleep deprivation """
    paths = [
        "/data/Clustering/SleepDeprivation/RatJ/Day2/",
        "/data/Clustering/SleepDeprivation/RatK/Day2/",
        "/data/Clustering/SleepDeprivation/RatN/Day2/",
    ]
    if indx is not None:
        paths = [paths[_] for _ in indx]

    return [processData(_) for _ in paths]


def two_novel():
    paths = [
        "/data/Clustering/SleepDeprivation/RatJ/Day3/",
        "/data/Clustering/SleepDeprivation/RatK/Day3/",
        "/data/Clustering/SleepDeprivation/RatN/Day3/",
    ]
    return [processData(_) for _ in paths]


def openfield():
    paths = [
        "/data/Clustering/SleepDeprivation/RatJ/Day4/",
        "/data/Clustering/SleepDeprivation/RatK/Day4/",
        "/data/Clustering/SleepDeprivation/RatN/Day4/",
    ]
    return [processData(_) for _ in paths]
