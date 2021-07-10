from neuropy.io import NeuroscopeIO, BinarysignalIO
from neuropy import core
from pathlib import Path
import numpy as np


def file_loader(f):
    return np.load(f, allow_pickle=True).item()


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

        # self.pf1d = sessobj.PF1d(self.recinfo)
        # self.pf2d = sessobj.PF2d(self.recinfo)
        # self.decode1D = sessobj.Decode1d(self.pf1d)
        # self.decode2D = sessobj.Decode2d(self.pf2d)
        # self.localsleep = sessobj.LocalSleep(self.recinfo)
        # self.pbe = sessobj.Pbe(self.recinfo)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.recinfo.session.sessionName})"
