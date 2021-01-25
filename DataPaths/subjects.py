from callfunc import processData
from typing import List

colors = {"sd": "#ff6b6b", "nsd": "#69c"}


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


class Openfield:
    @property
    def ratJday4(self):
        path = "/data/Clustering/SleepDeprivation/RatJ/Day4/"
        return [processData(path)]

    @property
    def ratNday4(self):
        path = "/data/Clustering/SleepDeprivation/RatN/Day4/"
        return [processData(path)]


class Sd:
    @property
    def allsess(self):
        pipelines: List[processData]
        pipelines = self.ratJday1 + self.ratKday1 + self.ratNday1 + self.ratSday3
        return pipelines

    @property
    def ratJday1(self):
        path = "/data/Clustering/SleepDeprivation/RatJ/Day1/"
        return [processData(path)]

    @property
    def ratKday1(self):
        path = "/data/Clustering/SleepDeprivation/RatK/Day1/"
        return [processData(path)]

    @property
    def ratNday1(self):
        path = "/data/Clustering/SleepDeprivation/RatN/Day1/"
        return [processData(path)]

    @property
    def ratSday3(self):
        path = "/data/Clustering/SleepDeprivation/RatS/Day3SD/"
        return [processData(path)]

    def __add__(self, other):
        pipelines: List[processData] = self.allsess + other.allsess
        return pipelines


class Nsd:
    @property
    def allsess(self):
        pipelines: List[processData]
        pipelines = self.ratJday2 + self.ratKday2 + self.ratNday2 + self.ratSday2
        return pipelines

    @property
    def ratJday2(self):
        path = "/data/Clustering/SleepDeprivation/RatJ/Day2/"
        return [processData(path)]

    @property
    def ratKday2(self):
        path = "/data/Clustering/SleepDeprivation/RatK/Day2/"
        return [processData(path)]

    @property
    def ratNday2(self):
        path = "/data/Clustering/SleepDeprivation/RatN/Day2/"
        return [processData(path)]

    @property
    def ratSday2(self):
        path = "/data/Clustering/SleepDeprivation/RatS/Day2NSD/"
        return [processData(path)]

    def __add__(self, other):
        pipelines: List[processData] = self.allsess + other.allsess
        return pipelines


class Two_novel:
    paths = [
        "/data/Clustering/SleepDeprivation/RatJ/Day3/",
        "/data/Clustering/SleepDeprivation/RatK/Day3/",
        "/data/Clustering/SleepDeprivation/RatN/Day3/",
    ]

    @property
    def ratSday5(self):
        path = "/data/Clustering/SleepDeprivation/RatS/Day5TwoNovel/"
        return [processData(path)]


# def sd(indx=None):
#     """Sleep deprivation sessions"""

#     paths = [
#         "/data/Clustering/SleepDeprivation/RatJ/Day1/",
#         "/data/Clustering/SleepDeprivation/RatK/Day1/",
#         "/data/Clustering/SleepDeprivation/RatN/Day1/",
#         "/data/Clustering/SleepDeprivation/RatS/Day3SD/",
#     ]
#     if indx is not None:
#         paths = [paths[_] for _ in indx]
#     return [processData(_) for _ in paths]


# def nsd(indx=None):
#     """Control sessions for sleep deprivation """
#     paths = [
#         "/data/Clustering/SleepDeprivation/RatJ/Day2/",
#         "/data/Clustering/SleepDeprivation/RatK/Day2/",
#         "/data/Clustering/SleepDeprivation/RatN/Day2/",
#     ]
#     if indx is not None:
#         paths = [paths[_] for _ in indx]

#     return [processData(_) for _ in paths]
