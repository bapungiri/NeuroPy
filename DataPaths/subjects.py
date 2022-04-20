from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from neuropy import core
from neuropy.io import BinarysignalIO, NeuroscopeIO
from scipy.ndimage import gaussian_gradient_magnitude

lineplot_kw = dict(
    marker="o",
    err_style="bars",
    linewidth=1,
    legend=None,
    mew=0.2,
    markersize=2,
    err_kws=dict(elinewidth=1, zorder=-1, capsize=1),
)

errorbar_kw = dict(
    marker="o",
    capsize=1,
    elinewidth=1,
    mec="w",
    markersize=2,
    linewidth=1,
    mew=0.2,
)


def boxplot_kw(color, lw=1):
    return dict(
        showfliers=False,
        linewidth=lw,
        boxprops=dict(facecolor="none", edgecolor=color),
        showcaps=True,
        capprops=dict(color=color),
        medianprops=dict(color=color, lw=lw),
        whiskerprops=dict(color=color),
    )


def light_cycle_span(ax, dark_start=-4.2, light_stop=9, dark_stop=0, light_start=0):
    ax.axvspan(dark_start, dark_stop, 0, 0.05, color="#6d6d69")
    ax.axvspan(light_start, light_stop, 0, 0.05, color="#e6e6a2")


def epoch_span(ax, starts=(-4, -1, 0), stops=(-1, 0, 9), ymin=0.2, ymax=0.25, zorder=0):
    kw = dict(ymin=ymin, ymax=ymax, alpha=0.5, zorder=zorder, ec=None)
    labels, colors = ["PRE", "MAZE", "POST"], ["#cdd0cd", "#68c563", "#cdd0cd"]
    for start, stop, l, c in zip(starts, stops, labels, colors):
        ax.axvspan(start, stop, color=c, **kw)
        ax.text((stop - start) / 2, -0.52, l, fontsize=8)


def adjust_lightness(color, amount=0.5):
    import colorsys

    import matplotlib.colors as mc

    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    c = colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])
    return mc.to_hex(c)


fig_folder = Path("/home/bapung/Documents/figures/")
figpath_sd = Path(
    "/home/bapung/Dropbox (University of Michigan)/figures/sleep_deprivation"
)


def colors_sd(amount=1):
    # return ['#9575CD', '#FF80AB']
    # return ["#9575CD", "#FF9100"]
    return [Nsd.color(amount), Sd.color(amount)]


colors_sleep = {
    "AW": "#474343",
    "QW": "#b6afaf",
    "REM": "#eb9494",
    "NREM": "#667cfa",
}

colors_sleep_old = {
    "active": "#474343",
    "quiet": "#b6afaf",
    "rem": "#eb9494",
    "nrem": "#667cfa",
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

        if (f := self.filePrefix.with_suffix(".animal.npy")).is_file():
            d = np.load(f, allow_pickle=True).item()
            self.animal = core.Animal.from_dict(d)

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
        self.sw = core.Epoch.from_file(fp.with_suffix(".sw.npy"))
        self.spindle = core.Epoch.from_file(fp.with_suffix(".spindle.npy"))
        self.ripple = core.Epoch.from_file(fp.with_suffix(".ripple.npy"))
        self.theta = core.Epoch.from_file(fp.with_suffix(".theta.npy"))
        self.theta_epochs = core.Epoch.from_file(fp.with_suffix(".theta.epochs.npy"))
        self.pbe = core.Epoch.from_file(fp.with_suffix(".pbe.npy"))
        self.off = core.Epoch.from_file(fp.with_suffix(".off.npy"))

        self.maze_run = core.Epoch.from_file(fp.with_suffix(".maze.running.npy"))
        self.remaze_run = core.Epoch.from_file(fp.with_suffix(".remaze.running.npy"))

        # ---- neurons related ------------

        if (f := self.filePrefix.with_suffix(".neurons.iso.npy")).is_file():
            d = np.load(f, allow_pickle=True).item()
            self.neurons_iso = core.Neurons.from_dict(d)

        if (f := self.filePrefix.with_suffix(".position.npy")).is_file():
            d = np.load(f, allow_pickle=True).item()
            self.position = core.Position.from_dict(d)

        if (f := self.filePrefix.with_suffix(".maze.linear.npy")).is_file():
            d = np.load(f, allow_pickle=True).item()
            self.maze = core.Position.from_dict(d)

        if (f := self.filePrefix.with_suffix(".re-maze.linear.npy")).is_file():
            d = np.load(f, allow_pickle=True).item()
            self.remaze = core.Position.from_dict(d)

        if (f := self.filePrefix.with_suffix(".maze1.linear.npy")).is_file():
            d = np.load(f, allow_pickle=True).item()
            self.maze1 = core.Position.from_dict(d)

        if (f := self.filePrefix.with_suffix(".maze2.linear.npy")).is_file():
            d = np.load(f, allow_pickle=True).item()
            self.maze2 = core.Position.from_dict(d)

    @property
    def replay_pbe(self):
        if (f := self.filePrefix.with_suffix(".replay_pbe.npy")).is_file():
            d = np.load(f, allow_pickle=True).item()
            return core.Epoch.from_dict(d)

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

    @property
    def data_table(self):
        files = [
            "paradigm",
            "artifact",
            "brainstates",
            "spindle",
            "ripple",
            "theta",
            "pbe",
            "neurons",
            "position",
            "maze.linear",
            "re-maze.linear",
            "maze1.linear",
            "maze2.linear",
        ]

        df = pd.DataFrame(columns=files)
        is_exist = []
        for file in files:
            if self.filePrefix.with_suffix(f".{file}.npy").is_file():
                is_exist.append(True)
            else:
                is_exist.append(False)

        df.loc[0] = is_exist
        df.insert(0, "session", self.filePrefix.name)

        return df

    def save_data(d, f):
        np.save(f, arr=d)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.recinfo.source_file.name})\n"


def data_table(sessions: list):

    df = []
    for sess in sessions:
        df.append(sess.data_table)

    return pd.concat(df, ignore_index=True)


class Group:
    tag = None
    basedir = Path("/data/Clustering/sessions/")

    def _process(self, rel_path):
        return [ProcessData(self.basedir / rel_path, self.tag)]

    def data_exist(self):
        self.allsess


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
    tag = "SD"

    @property
    def allsess(self):
        pipelines: List[ProcessData]
        pipelines = (
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
    def mua_sess(self):
        pipelines: List[ProcessData]
        pipelines = (
            self.ratJday1
            + self.ratKday1
            + self.ratNday1
            + self.ratSday3
            + self.ratRday2
            + self.ratUday4
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
    def brainstates_sess(self):
        pipelines: List[ProcessData] = (
            self.ratJday1
            + self.ratKday1
            + self.ratNday1
            + self.ratSday3
            + self.ratRday2
            + self.ratUday4
        )
        return pipelines

    @property
    def pf_sess(self):
        pipelines: List[ProcessData] = (
            self.ratKday1
            + self.ratNday1
            + self.ratSday3
            + self.ratUday1
            + self.ratUday4
            + self.ratVday2
            + self.ratRday2
        )
        return pipelines

    @property
    def remaze(self):
        pipelines: List[ProcessData] = (
            self.ratSday3
            + self.ratUday1
            + self.ratUday4
            + self.ratVday2
            + self.ratRday2
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
        # return adjust_lightness("#df670c", amount=amount)
        # return adjust_lightness("#f06292", amount=amount)
        return adjust_lightness("#ff0000", amount=amount)

    @staticmethod
    def rs_color(amount=1):
        return adjust_lightness("#00B8D4", amount=amount)


class Nsd(Group):
    tag = "NSD"

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
            + self.ratVday3
        )
        return pipelines

    @property
    def mua_sess(self):
        pipelines: List[ProcessData]
        pipelines = (
            self.ratJday2
            + self.ratKday2
            + self.ratNday2
            + self.ratSday2
            + self.ratUday2
            + self.ratVday1
            + self.ratVday3
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
    def brainstates_sess(self):
        pipelines: List[ProcessData] = (
            self.ratJday2
            + self.ratKday2
            + self.ratNday2
            + self.ratSday2
            + self.ratUday2
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
            + self.ratVday3
        )
        return pipelines

    @property
    def remaze(self):
        pipelines: List[ProcessData] = (
            self.ratSday2 + self.ratUday2 + self.ratVday1 + self.ratVday3
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
        # return adjust_lightness("#815bcd", amount=amount)
        return adjust_lightness("#424242", amount=amount)


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
    __slots__ = (
        "path",
        "swa_examples",
        "brainstates_proportion",
        "ripple_psd",
        "ripple_examples" "ripple_rate",
        "ripple_total_duration",
        "ripple_peak_frequency",
        "ripple_zscore",
        "ripple_autocorr",
        "pbe_rate",
        "pbe_total_duration",
        "frate_zscore",
        "frate_interneuron_around_Zt5",
        "frate_change_1vs5",
        "frate_change_pre_to_post",
        "frate_pre_to_maze_quantiles_in_POST",
        "frate_pre_to_maze_quantiles_in_POST_shuffled",
        "frate_in_ripple",
        "ev_pooled",
        "ev_brainstates",
        "pf_norm_tuning",
        "replay_examples",
        "replay_sig_frames",
        "replay_wcorr",
        "replay_re_maze_score",
        "replay_post_score",
        "replay_pos_distribution",
        "replay_re_maze_position_distribution",
        "remaze_ev_example",
        "remaze_ev_on_POST_example",
        "remaze_ev",
        "remaze_temporal_bias",
        "remaze_maze_paircorr",
        "remaze_first5_paircorr",
        "remaze_first5_subsample",
        "remaze_first5_bootstrap",
        "remaze_last5_paircorr",
        "remaze_corr_across_session",
        "remaze_activation_of_maze",
        "remaze_temporal_bias_com_correlation_across_session",
        "remaze_ensemble_corr_across_sess",
        "remaze_ensemble_activation_across_sess",
        "remaze_ev_on_zt0to5",
        "remaze_ev_on_POST_pooled",
        "post_first5_last5_paircorr",
    )

    def __init__(self) -> None:
        self.path = Path("/home/bapung/Dropbox (University of Michigan)/ProcessedData")
        # for f in self.path.iterdir():
        #     setattr(self, f.name, self.load(f.stem))

    def save(self, d, fp):
        data = {"data": d}
        np.save(self.path / fp, data)
        print(f"{fp} saved")

    def load(self, fp):
        return np.load(self.path / f"{fp}.npy", allow_pickle=True).item()

    def __getattr__(self, name: str):
        return self.load(name)["data"]


sd = Sd()
nsd = Nsd()
of = Of()
tn = Tn()


def mua_sess():
    return nsd.mua_sess + sd.mua_sess


def pf_sess():
    return nsd.pf_sess + sd.pf_sess


def ripple_sess():
    return nsd.ripple_sess + sd.ripple_sess
