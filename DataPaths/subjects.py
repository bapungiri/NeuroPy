from typing import List
from neuropy.io import NeuroscopeIO, BinarysignalIO
from neuropy import core
from pathlib import Path
import numpy as np


def sd_span(ax, s=2, w=2, h=0.05):
    vis = {"alpha": 0.5, "zorder": 0, "ec": None}
    sd_col, rs_col = Sd.color(1), Sd.rs_color(1)
    ax.axvspan(s, s + w, 0, h, color=sd_col, **vis)
    ax.axvspan(s + w, s + 2 * w, 0, h, color=rs_col, **vis)


def nsd_span(ax, s=2, w=2, h=0.05):
    vis = {"alpha": 0.5, "zorder": 0, "ec": None}
    nsd_col = Nsd.color(1)
    ax.axvspan(s, s + 2 * w, h, 2 * h, color=nsd_col, **vis)


def grp_span(ax, s=2, w=2, h=0.05):
    vis = {"alpha": 0.5, "zorder": 0, "ec": None}
    sd_col, rs_col = Sd.color(1), Sd.rs_color(1)
    nsd_col = Nsd.color(1)
    ax.axvspan(s, s + w, 0, h, color=sd_col, **vis)
    ax.axvspan(s + w, s + 2 * w, 0, h, color=rs_col, **vis)
    ax.axvspan(s, s + 2 * w, h + 0.001, 2 * h, color=nsd_col, **vis)


def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys

    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    c = colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])
    return mc.to_hex(c)


def file_loader(f):
    return np.load(f, allow_pickle=True).item()


fig_folder = Path("/home/bapung/Documents/figures/")
figpath_sd = Path(
    "/home/bapung/Dropbox (University of Michigan)/figures/sleep_deprivation"
)


def colors_sd(amount=1):
    return [Nsd.color(amount), Sd.color(amount)]


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
        self.off = core.Epoch.from_file(fp.with_suffix(".off.npy"))

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
            self.maze = core.Position.from_dict(d)

        if (f := self.filePrefix.with_suffix(".re-maze.linear.npy")).is_file():
            d = np.load(f, allow_pickle=True).item()
            self.re_maze = core.Position.from_dict(d)

        if (f := self.filePrefix.with_suffix(".maze1.linear.npy")).is_file():
            d = np.load(f, allow_pickle=True).item()
            self.maze1 = core.Position.from_dict(d)

        if (f := self.filePrefix.with_suffix(".maze2.linear.npy")).is_file():
            d = np.load(f, allow_pickle=True).item()
            self.maze2 = core.Position.from_dict(d)

    @property
    def neurons(self):
        # it is relatively heavy on memory hence loaded only while required
        if (f := self.filePrefix.with_suffix(".neurons.npy")).is_file():
            d = np.load(f, allow_pickle=True).item()
            return core.Neurons.from_dict(d)

    @property
    def mua(self):
        if (f := self.filePrefix.with_suffix(".mua.npy")).is_file():
            d = np.load(f, allow_pickle=True).item()
            return core.Mua.from_dict(d)

    def save_data(d, f):
        np.save(f, arr=d)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.recinfo.source_file.name})\n"


def allsess():
    """all data folders together"""
    paths = [
        "RatJ/Day1/",
        "RatK/Day1/",
        "RatN/Day1/",
        "RatJ/Day2/",
        "RatK/Day2/",
        "RatN/Day2/",
        "RatJ/Day3/",
        "RatK/Day3/",
        "RatN/Day3/",
        "RatJ/Day4/",
        "RatK/Day4/",
        "RatN/Day4/",
        "RatA14d1LP/Rollipram/",
    ]
    return [ProcessData(_) for _ in paths]


class Group:
    tag = None
    basedir = Path("/data/Clustering/sessions/")

    def _process(self, rel_path):
        return [ProcessData(self.basedir / rel_path, self.tag)]


class Of:
    @property
    def ratJday4(self):
        path = "RatJ/Day4/"
        return [ProcessData(path)]

    @property
    def ratKday4(self):
        path = "RatK/Day4/"
        return [ProcessData(path)]

    @property
    def ratNday4(self):
        path = "RatN/Day4/"
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
    def ripple_sess(self):
        pipelines: List[ProcessData] = (
            self.ratJday1
            + self.ratKday1
            + self.ratNday1
            + self.ratSday3
            + self.ratRday2
            + self.ratUday1
            + self.ratUday4
            + self.ratVday2
        )
        return pipelines

    @property
    def pf_sess(self):
        pipelines: List[ProcessData] = (
            self.ratKday1
            + self.ratNday1
            + self.ratSday3
            + self.ratUday4
            + self.ratVday2
        )
        return pipelines

    @property
    def ratJday1(self):
        return self._process("RatJ/Day1/")

    @property
    def ratKday1(self):
        return self._process("RatK/Day1/")

    @property
    def ratNday1(self):
        return self._process("RatN/Day1/")

    @property
    def ratSday3(self):
        return self._process("RatS/Day3SD/")

    @property
    def ratRday2(self):
        return self._process("RatR/Day2SD")

    @property
    def ratUday1(self):
        return self._process("RatU/RatUDay1SD")

    @property
    def ratUday4(self):
        return self._process("RatU/RatUDay4SD")

    @property
    def ratVday2(self):
        return self._process("RatV/RatVDay2SD/")

    # @property
    # def ratUday5(self):
    #     path = "/data/Clustering/sessions/RatU/RatUDay5OpenfieldSD/"
    #     return [ProcessData(path)]

    @property
    def utkuAG_day1(self):
        path = "Utku/AG_2019-12-22_SD_day1/"
        return [ProcessData(path)]

    @property
    def utkuAG_day2(self):
        path = "Utku/AG_2019-12-26_SD_day2/"
        return [ProcessData(path)]

    def __add__(self, other):
        pipelines: List[ProcessData] = self.allsess + other.allsess
        return pipelines

    @staticmethod
    def color(amount=1):
        return adjust_lightness("#df670c", amount=amount)

    @staticmethod
    def rs_color(amount=1):
        return adjust_lightness("#00B8D4", amount=amount)


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
    def ripple_sess(self):
        pipelines: List[ProcessData] = (
            self.ratJday2
            + self.ratKday2
            + self.ratNday2
            + self.ratSday2
            + self.ratRday1
            + self.ratUday2
            + self.ratVday1
            + self.ratVday3
        )
        return pipelines

    @property
    def pf_sess(self):
        pipelines: List[ProcessData] = (
            self.ratKday2
            + self.ratNday2
            + self.ratSday2
            + self.ratUday2
            + self.ratVday1
        )
        return pipelines

    @property
    def ratJday2(self):
        return self._process("RatJ/Day2/")

    @property
    def ratKday2(self):
        return self._process("RatK/Day2/")

    @property
    def ratNday2(self):
        return self._process("RatN/Day2/")

    @property
    def ratSday2(self):
        return self._process("RatS/Day2NSD/")

    @property
    def ratRday1(self):
        return self._process("RatR/Day1NSD/")

    @property
    def ratUday2(self):
        return self._process("RatU/RatUDay2NSD/")

    @property
    def ratVday1(self):
        return self._process("RatV/RatVDay1NSD/")

    @property
    def ratVday3(self):
        return self._process("RatV/RatVDay3NSD")

    def __add__(self, other):
        pipelines: List[ProcessData] = self.allsess + other.allsess
        return pipelines

    @staticmethod
    def color(amount=1):
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
    def ripple_total_duration(self):
        return self.load("ripple_total_duration")["data"]

    @property
    def ripple_peak_frequency(self):
        return self.load("ripple_peak_frequency")["data"]

    @property
    def pbe_rate(self):
        return self.load("pbe_rate")["data"]

    @property
    def pbe_total_duration(self):
        return self.load("pbe_total_duration")["data"]

    @property
    def frate_pyr_in_ripple(self):
        return self.load("frate_pyr_in_ripple")["data"]

    @property
    def frate_inter_in_ripple(self):
        return self.load("frate_inter_in_ripple")["data"]

    @property
    def ev_pooled(self):
        return self.load("ev_pooled")["data"]

    @property
    def pf_norm_tuning(self):
        return self.load("pf_norm_tuning")["data"]


sd = Sd()
nsd = Nsd()
of = Of()
tn = Tn()
