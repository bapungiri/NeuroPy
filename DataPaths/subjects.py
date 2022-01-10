from typing import List
from neuropy.io import NeuroscopeIO, BinarysignalIO
from neuropy import core
from pathlib import Path
import numpy as np


def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys

    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def file_loader(f):
    return np.load(f, allow_pickle=True).item()


fig_folder = Path("/home/bapung/Documents/figures/")
figpath_sd = Path(
    "/home/bapung/Dropbox (University of Michigan)/figures/sleep_deprivation"
)

# sd_colors = {"sd": "#ff6b6b", "nsd": "#69c"}
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
    def __init__(self, basepath, tag=None):
        basepath = Path(basepath)
        xml_files = sorted(basepath.glob("*.xml"))
        assert len(xml_files) == 1, "Found more than one .xml file"

        fp = xml_files[0].with_suffix("")
        self.filePrefix = fp
        self.sub_name = fp.name[:4]
        self.tag = tag

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
        # epoch_names = [
        #     "paradigm",
        #     "artifact",
        #     "brainstates",
        #     "spindle",
        #     "ripple",
        #     "theta",
        #     "pbe",
        # ]
        # for e in epoch_names:
        #     setattr(self, e, core.Epoch.from_file(fp.with_suffix(f".{e}.npy")))
        self.paradigm = core.Epoch.from_file(fp.with_suffix(".paradigm.npy"))
        self.artifact = core.Epoch.from_file(fp.with_suffix(".artifact.npy"))
        self.brainstates = core.Epoch.from_file(fp.with_suffix(".brainstates.npy"))
        self.spindle = core.Epoch.from_file(fp.with_suffix(".spindle.npy"))
        self.ripple = core.Epoch.from_file(fp.with_suffix(".ripple.npy"))
        self.theta = core.Epoch.from_file(fp.with_suffix(".theta.npy"))
        self.theta_epochs = core.Epoch.from_file(fp.with_suffix(".theta.epochs.npy"))
        self.pbe = core.Epoch.from_file(fp.with_suffix(".pbe.npy"))

        # self.mua = core.Mua.from_file(fp.with_suffix(".mua.npy"))
        # self.position = core.Position(
        #     filename=self.filePrefix.with_suffix(".position.npy")
        # )

        # ---- neurons related ------------

        if (f := self.filePrefix.with_suffix(".running.npy")).is_file():
            d = np.load(f, allow_pickle=True).item()
            self.run = core.Epoch.from_dict(d)

        if (f := self.filePrefix.with_suffix(".replay.pbe.npy")).is_file():
            d = np.load(f, allow_pickle=True).item()
            self.replay_pbe = core.Epoch.from_dict(d)

        # if (f := self.filePrefix.with_suffix(".neurons.npy")).is_file():
        #     d = np.load(f, allow_pickle=True).item()
        #     self.neurons = core.Neurons.from_dict(d)

        if (f := self.filePrefix.with_suffix(".neurons.iso.npy")).is_file():
            d = np.load(f, allow_pickle=True).item()
            self.neurons_iso = core.Neurons.from_dict(d)

        # if (f := self.filePrefix.with_suffix(".mua.npy")).is_file():
        #     d = np.load(f, allow_pickle=True).item()
        #     self.mua = core.Mua.from_dict(d)

        if (f := self.filePrefix.with_suffix(".position.npy")).is_file():
            d = np.load(f, allow_pickle=True).item()
            self.position = core.Position.from_dict(d)

        if (f := self.filePrefix.with_suffix(".maze.linear.npy")).is_file():
            d = np.load(f, allow_pickle=True).item()
            self.lin_maze = core.Position.from_dict(d)

        if (f := self.filePrefix.with_suffix(".maze1.linear.npy")).is_file():
            d = np.load(f, allow_pickle=True).item()
            self.maze1 = core.Position.from_dict(d)

        if (f := self.filePrefix.with_suffix(".maze2.linear.npy")).is_file():
            d = np.load(f, allow_pickle=True).item()
            self.maze2 = core.Position.from_dict(d)

        # self.pf1d = sessobj.PF1d(self.recinfo)
        # self.pf2d = sessobj.PF2d(self.recinfo)
        # self.decode1D = sessobj.Decode1d(self.pf1d)
        # self.decode2D = sessobj.Decode2d(self.pf2d)
        # self.localsleep = sessobj.LocalSleep(self.recinfo)
        # self.pbe = sessobj.Pbe(self.recinfo)

    @property
    def neurons(self):
        # it is relatively heavy on memory hence loaded only while required
        if (f := self.filePrefix.with_suffix(".neurons.npy")).is_file():
            d = np.load(f, allow_pickle=True).item()
            return core.Neurons.from_dict(d)

    @property
    def mua(self):
        return core.Mua.from_file(self.filePrefix.with_suffix(".mua.npy"))

    def save_data(d, f):
        np.save(f, arr=d)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.recinfo.source_file.name})\n"


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


class Group:
    tag = None

    def _process(self, path):
        return [ProcessData(path, self.tag)]


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


class Sd(Group):
    tag = "sd"

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
            + self.ratVday2
        )
        return pipelines

    @property
    def ratJday1(self):
        return self._process("/data/Clustering/sessions/RatJ/Day1/")

    @property
    def ratKday1(self):
        return self._process("/data/Clustering/sessions/RatK/Day1/")

    @property
    def ratNday1(self):
        return self._process("/data/Clustering/sessions/RatN/Day1/")

    @property
    def ratSday3(self):
        return self._process("/data/Clustering/sessions/RatS/Day3SD/")

    @property
    def ratRday2(self):
        return self._process("/data/Clustering/sessions/RatR/Day2SD")

    @property
    def ratUday4(self):
        return self._process("/data/Clustering/sessions/RatU/RatUDay4SD")

    @property
    def ratVday2(self):
        return self._process("/data/Clustering/sessions/RatV/RatVDay2SD/")

    # @property
    # def ratUday5(self):
    #     path = "/data/Clustering/sessions/RatU/RatUDay5OpenfieldSD/"
    #     return [ProcessData(path)]

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

    def color(self, amount=1):
        return adjust_lightness("#df670c", amount=amount)


class Nsd(Group):
    tag = "nsd"

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
            + self.ratVday1
        )
        return pipelines

    @property
    def ratJday2(self):
        return self._process("/data/Clustering/sessions/RatJ/Day2/")

    @property
    def ratKday2(self):
        return self._process("/data/Clustering/sessions/RatK/Day2/")

    @property
    def ratNday2(self):
        return self._process("/data/Clustering/sessions/RatN/Day2/")

    @property
    def ratSday2(self):
        return self._process("/data/Clustering/sessions/RatS/Day2NSD/")

    @property
    def ratRday1(self):
        return self._process("/data/Clustering/sessions/RatR/Day1NSD/")

    @property
    def ratUday2(self):
        return self._process("/data/Clustering/sessions/RatU/RatUDay2NSD/")

    @property
    def ratVday1(self):
        return self._process("/data/Clustering/sessions/RatV/RatVDay1NSD/")

    def __add__(self, other):
        pipelines: List[ProcessData] = self.allsess + other.allsess
        return pipelines

    def color(self, amount=1):
        return adjust_lightness("#633bb5", amount=amount)


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


class GroupData:
    def __init__(self) -> None:
        self.path = Path("/home/bapung/Dropbox (University of Michigan)/ProcessedData")

    def save(self, d, fp):
        data = {"data": d}
        np.save(self.path / fp, data)

    def load(self, fp):
        return np.load(self.path / f"{fp}.npy", allow_pickle=True).item()

    @property
    def ripple_psd(self):
        return self.load("ripple_psd")["data"]

    @property
    def ripple_rate(self):
        return self.load("ripple_rate")["data"]

    @property
    def ripple_peak_frequency(self):
        return self.load("ripple_peak_frequency")["data"]

    @property
    def ev_pooled(self):
        return self.load("ev_pooled")["data"]
