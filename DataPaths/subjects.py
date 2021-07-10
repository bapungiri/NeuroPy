from neuropy_loader import ProcessData
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
    return [ProcessData(_) for _ in paths]


class Of:
    @property
    def ratJday4(self):
        path = "/data/Clustering/SleepDeprivation/RatJ/Day4/"
        return [ProcessData(path)]

    @property
    def ratKday4(self):
        path = "/data/Clustering/SleepDeprivation/RatK/Day4/"
        return [ProcessData(path)]

    @property
    def ratNday4(self):
        path = "/data/Clustering/SleepDeprivation/RatN/Day4/"
        return [ProcessData(path)]


class Sd:
    @property
    def allsess(self):
        pipelines: List[ProcessData]
        pipelines = (
            self.ratJday1
            + self.ratKday1
            + self.ratNday1
            + self.ratSday3
            + self.ratRday2
        )
        return pipelines

    @property
    def ratJday1(self):
        path = "/data/Clustering/SleepDeprivation/RatJ/Day1/"
        return [ProcessData(path)]

    @property
    def ratKday1(self):
        path = "/data/Clustering/SleepDeprivation/RatK/Day1/"
        return [ProcessData(path)]

    @property
    def ratNday1(self):
        path = "/data/Clustering/SleepDeprivation/RatN/Day1/"
        return [ProcessData(path)]

    @property
    def ratSday3(self):
        path = "/data/Clustering/SleepDeprivation/RatS/Day3SD/"
        return [ProcessData(path)]

    @property
    def ratRday2(self):
        path = "/data/Clustering/SleepDeprivation/RatR/Day2SD"
        return [ProcessData(path)]

    @property
    def utkuAG_day1(self):
        path = "/data/Clustering/SleepDeprivation/Utku/AG_2019-12-22_SD_day1/"
        return [ProcessData(path)]

    @property
    def utkuAG_day2(self):
        path = "/data/Clustering/SleepDeprivation/Utku/AG_2019-12-26_SD_day2/"
        return [ProcessData(path)]

    def __add__(self, other):
        pipelines: List[ProcessData] = self.allsess + other.allsess
        return pipelines


class Nsd:
    @property
    def allsess(self):
        pipelines: List[ProcessData]
        pipelines = self.ratJday2 + self.ratKday2 + self.ratNday2 + self.ratSday2
        return pipelines

    @property
    def ratJday2(self):
        path = "/data/Clustering/SleepDeprivation/RatJ/Day2/"
        return [ProcessData(path)]

    @property
    def ratKday2(self):
        path = "/data/Clustering/SleepDeprivation/RatK/Day2/"
        return [ProcessData(path)]

    @property
    def ratNday2(self):
        path = "/data/Clustering/SleepDeprivation/RatN/Day2/"
        return [ProcessData(path)]

    @property
    def ratSday2(self):
        path = "/data/Clustering/SleepDeprivation/RatS/Day2NSD/"
        return [ProcessData(path)]

    @property
    def ratRday1(self):
        path = "/data/Clustering/SleepDeprivation/RatR/Day1NSD/"
        return [ProcessData(path)]

    def __add__(self, other):
        pipelines: List[ProcessData] = self.allsess + other.allsess
        return pipelines


class Tn:
    paths = [
        "/data/Clustering/SleepDeprivation/RatJ/Day3/",
        "/data/Clustering/SleepDeprivation/RatK/Day3/",
        "/data/Clustering/SleepDeprivation/RatN/Day3/",
    ]

    @property
    def ratSday5(self):
        path = "/data/Clustering/SleepDeprivation/RatS/Day5TwoNovel/"
        return [ProcessData(path)]


sd = Sd()
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
#     return [ProcessData(_) for _ in paths]


# def nsd(indx=None):
#     """Control sessions for sleep deprivation """
#     paths = [
#         "/data/Clustering/SleepDeprivation/RatJ/Day2/",
#         "/data/Clustering/SleepDeprivation/RatK/Day2/",
#         "/data/Clustering/SleepDeprivation/RatN/Day2/",
#     ]
#     if indx is not None:
#         paths = [paths[_] for _ in indx]

#     return [ProcessData(_) for _ in paths]
