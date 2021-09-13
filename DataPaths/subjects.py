from typing import List
from neuropy.io import NeuroscopeIO, BinarysignalIO
from neuropy import core
from pathlib import Path
import numpy as np


def file_loader(f):
    return np.load(f, allow_pickle=True).item()


fig_folder = Path("/home/bapung/Documents/figures/")
figpath_sd = Path("/home/bapung/Documents/figures/sleep_deprivation")

# sd_colors = {"sd": "#ff6b6b", "nsd": "#69c"}
sd_colors = {"sd": "#df670c", "nsd": "#633bb5"}
sleep_colors = {
    "nrem": "#667cfa",
    "rem": "#eb9494",
    "quiet": "#b6afaf",
    "active": "#474343",
}

# sleep_colors = {
#     "nrem": "#7a13c3",
#     "rem": "#08af5f",
#     "quiet": "#d67105",
#     "active": "#b20a50",
# }


class ProcessData:
    def __init__(self, basepath):
        basepath = Path(basepath)
        xml_files = sorted(basepath.glob("*.xml"))
        assert len(xml_files) == 1, "Found more than one .xml file"

        fp = xml_files[0].with_suffix("")
        self.filePrefix = fp

        self.recinfo = NeuroscopeIO(xml_files[0])
        self.eegfile = BinarysignalIO(
            self.recinfo.eeg_filename,
            n_channels=self.recinfo.n_channels,
            sampling_rate=self.recinfo.eeg_sampling_rate,
        )

        if self.recinfo.dat_filename.is_file():
            self.datfile = BinarysignalIO(
                self.recinfo.dat_filename,
                n_channels=self.recinfo.n_channels,
                sampling_rate=self.recinfo.dat_sampling_rate,
            )

        self.probegroup = core.ProbeGroup.from_file(fp.with_suffix(".probegroup.npy"))

        # ----- epochs --------------
        self.paradigm = core.Epoch.from_file(fp.with_suffix(".paradigm.npy"))
        self.artifact = core.Epoch.from_file(fp.with_suffix(".artifact.npy"))
        self.brainstates = core.Epoch.from_file(fp.with_suffix(".brainstates.npy"))
        self.ripple = core.Epoch.from_file(fp.with_suffix(".ripple.npy"))
        self.theta = core.Epoch.from_file(fp.with_suffix(".theta.npy"))

        self.pbe = core.Epoch.from_file(fp.with_suffix(".pbe.npy"))
        self.mua = core.Mua.from_file(fp.with_suffix(".mua.npy"))
        # self.position = core.Position(
        #     filename=self.filePrefix.with_suffix(".position.npy")
        # )

        # ---- neurons related ------------

        if (f := self.filePrefix.with_suffix(".neurons.npy")).is_file():
            d = np.load(f, allow_pickle=True).item()
            self.neurons = core.Neurons.from_dict(d)

        if (f := self.filePrefix.with_suffix(".mua.npy")).is_file():
            d = np.load(f, allow_pickle=True).item()
            self.mua = core.Mua.from_dict(d)

        if (f := self.filePrefix.with_suffix(".position.npy")).is_file():
            d = np.load(f, allow_pickle=True).item()
            self.position = core.Position.from_dict(d)

        # self.pf1d = sessobj.PF1d(self.recinfo)
        # self.pf2d = sessobj.PF2d(self.recinfo)
        # self.decode1D = sessobj.Decode1d(self.pf1d)
        # self.decode2D = sessobj.Decode2d(self.pf2d)
        # self.localsleep = sessobj.LocalSleep(self.recinfo)
        # self.pbe = sessobj.Pbe(self.recinfo)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.recinfo.source_file.name})"


def allsess():
    """all data folders together"""
    paths = [
        "/data/Clustering/sessions/RatJ/Day1/",
        "/data/Clustering/sessions/RatK/Day1/",
        "/data/Clustering/sessions/RatN/Day1/",
        "/data/Clustering/sessions/RatJ/Day2/",
        "/data/Clustering/sessions/RatK/Day2/",
        "/data/Clustering/sessions/RatN/Day2/",
        "/data/Clustering/sessions/RatJ/Day3/",
        "/data/Clustering/sessions/RatK/Day3/",
        "/data/Clustering/sessions/RatN/Day3/",
        "/data/Clustering/sessions/RatJ/Day4/",
        "/data/Clustering/sessions/RatK/Day4/",
        "/data/Clustering/sessions/RatN/Day4/",
        "/data/Clustering/sessions/RatA14d1LP/Rollipram/",
    ]
    return [ProcessData(_) for _ in paths]


class Of:
    @property
    def ratJday4(self):
        path = "/data/Clustering/sessions/RatJ/Day4/"
        return [ProcessData(path)]

    @property
    def ratKday4(self):
        path = "/data/Clustering/sessions/RatK/Day4/"
        return [ProcessData(path)]

    @property
    def ratNday4(self):
        path = "/data/Clustering/sessions/RatN/Day4/"
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
            + self.ratUday4
            + self.ratUday5
        )
        return pipelines

    @property
    def ratJday1(self):
        path = "/data/Clustering/sessions/RatJ/Day1/"
        return [ProcessData(path)]

    @property
    def ratKday1(self):
        path = "/data/Clustering/sessions/RatK/Day1/"
        return [ProcessData(path)]

    @property
    def ratNday1(self):
        path = "/data/Clustering/sessions/RatN/Day1/"
        return [ProcessData(path)]

    @property
    def ratSday3(self):
        path = "/data/Clustering/sessions/RatS/Day3SD/"
        return [ProcessData(path)]

    @property
    def ratRday2(self):
        path = "/data/Clustering/sessions/RatR/Day2SD"
        return [ProcessData(path)]

    @property
    def ratUday4(self):
        path = "/data/Clustering/sessions/RatU/RatUDay4SD"
        return [ProcessData(path)]

    @property
    def ratUday5(self):
        path = "/data/Clustering/sessions/RatU/RatUDay5OpenfieldSD/"
        return [ProcessData(path)]

    @property
    def utkuAG_day1(self):
        path = "/data/Clustering/sessions/Utku/AG_2019-12-22_SD_day1/"
        return [ProcessData(path)]

    @property
    def utkuAG_day2(self):
        path = "/data/Clustering/sessions/Utku/AG_2019-12-26_SD_day2/"
        return [ProcessData(path)]

    def __add__(self, other):
        pipelines: List[ProcessData] = self.allsess + other.allsess
        return pipelines


class Nsd:
    @property
    def allsess(self):
        pipelines: List[ProcessData]
        pipelines = (
            self.ratJday2
            + self.ratKday2
            + self.ratNday2
            + self.ratSday2
            + self.ratRday1
            + self.ratUday2
        )
        return pipelines

    @property
    def ratJday2(self):
        path = "/data/Clustering/sessions/RatJ/Day2/"
        return [ProcessData(path)]

    @property
    def ratKday2(self):
        path = "/data/Clustering/sessions/RatK/Day2/"
        return [ProcessData(path)]

    @property
    def ratNday2(self):
        path = "/data/Clustering/sessions/RatN/Day2/"
        return [ProcessData(path)]

    @property
    def ratSday2(self):
        path = "/data/Clustering/sessions/RatS/Day2NSD/"
        return [ProcessData(path)]

    @property
    def ratRday1(self):
        path = "/data/Clustering/sessions/RatR/Day1NSD/"
        return [ProcessData(path)]

    @property
    def ratUday2(self):
        path = "/data/Clustering/sessions/RatU/RatUDay2NSD/"
        return [ProcessData(path)]

    def __add__(self, other):
        pipelines: List[ProcessData] = self.allsess + other.allsess
        return pipelines


class Tn:
    paths = [
        "/data/Clustering/sessions/RatJ/Day3/",
        "/data/Clustering/sessions/RatK/Day3/",
        "/data/Clustering/sessions/RatN/Day3/",
    ]

    @property
    def ratSday5(self):
        path = "/data/Clustering/sessions/RatS/Day5TwoNovel/"
        return [ProcessData(path)]


sd = Sd()
nsd = Nsd()
of = Of()
tn = Tn()
# def sd(indx=None):
#     """Sleep deprivation sessions"""

#     paths = [
#         "/data/Clustering/sessions/RatJ/Day1/",
#         "/data/Clustering/sessions/RatK/Day1/",
#         "/data/Clustering/sessions/RatN/Day1/",
#         "/data/Clustering/sessions/RatS/Day3SD/",
#     ]
#     if indx is not None:
#         paths = [paths[_] for _ in indx]
#     return [ProcessData(_) for _ in paths]


# def nsd(indx=None):
#     """Control sessions for sleep deprivation """
#     paths = [
#         "/data/Clustering/sessions/RatJ/Day2/",
#         "/data/Clustering/sessions/RatK/Day2/",
#         "/data/Clustering/sessions/RatN/Day2/",
#     ]
#     if indx is not None:
#         paths = [paths[_] for _ in indx]

#     return [ProcessData(_) for _ in paths]
