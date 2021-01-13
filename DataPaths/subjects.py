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


def openfield():
    paths = [
        "/data/Clustering/SleepDeprivation/RatJ/Day4/",
        "/data/Clustering/SleepDeprivation/RatK/Day4/",
        "/data/Clustering/SleepDeprivation/RatN/Day4/",
    ]
    return [processData(_) for _ in paths]


class Sd:
    @property
    def allsess(self):
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


class Nsd:
    @property
    def allsess(self):
        pipelines = self.ratJday2 + self.ratKday2 + self.ratNday2
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

    # @property
    # def ratSday2(self):
    #     path = "/data/Clustering/SleepDeprivation/RatS/Day2NSD/"
    #     return [processData(path)]


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
