import os
from pathlib import Path

import numpy as np

from lfpEvent import swr, hwsa
from makeChanMap import ExtractChanXml
from MakePrmKlusta import makePrmPrb


class name2path:
    # common parameters used frequently
    lfpsRate = 1250

    def __init__(self, basePath):
        self.sessionName = basePath.split("/")[-3] + basePath.split("/")[-2]
        self.name = basePath.split("/")[-3]
        self.day = basePath.split("/")[-2]
        self.basePath = Path(basePath)
        for file in os.listdir(basePath):
            if file.endswith(".xml"):
                self.xmlfile = self.basePath / file
                self.subname = self.xmlfile.stem
                self.filePrefix = self.xmlfile.with_suffix("")
                self.eegfile = self.xmlfile.with_suffix(".eeg")
                self.datfile = self.xmlfile.with_suffix(".dat")

        self.f_basics = Path(str(self.filePrefix) + "_basics.npy")
        self.f_ripplelfp = Path(str(self.filePrefix) + "_BestRippleChans.npy")
        self.f_thetalfp = Path(str(self.filePrefix) + "_BestThetaChan.npy")
        self.f_ripple_evt = Path(str(self.filePrefix) + "_ripples.npy")
        self.f_theta_evt = Path(str(self.filePrefix) + "_thetaevents.npy")
        self.f_sessionepoch = Path(str(self.filePrefix) + "_epochs.npy")

        # parameters used across


class session(ExtractChanXml, makePrmPrb, swr, hwsa):
    def __init__(self, basePath):
        super().__init__(basePath)
