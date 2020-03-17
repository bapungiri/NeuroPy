import os
from pathlib import Path

import numpy as np


class path2files:
    def __init__(self, basePath):
        basePath = Path(basePath)

        for file in os.listdir(basePath):
            if file.endswith(".xml"):
                xmlfile = basePath / file
                filePrefix = xmlfile.with_suffix("")
        self._session = sessionname(basePath)
        self._files = files(filePrefix)
        self._recfiles = recfiles(filePrefix)


class files:
    def __init__(self, f_prefix):

        self.basics = Path(str(f_prefix) + "_basics.npy")
        self.epochs = Path(str(f_prefix) + "_epochs.npy")
        self.ripplelfp = Path(str(f_prefix) + "_BestRippleChans.npy")
        self.ripple_evt = Path(str(f_prefix) + "_ripples.npy")

        self.thetalfp = Path(str(f_prefix) + "_BestThetaChan.npy")
        self.theta_evt = Path(str(f_prefix) + "_thetaevents.npy")
        self.sessionepoch = Path(str(f_prefix) + "_epochs.npy")
        self.hwsa_ripple = Path(str(f_prefix) + "_hswa_ripple.npy")
        self.sws_states = Path(str(f_prefix) + "_sws.npy")
        self.slow_wave = Path(str(f_prefix) + "_hswa.npy")


class recfiles:
    def __init__(self, f_prefix):

        self.xmlfile = f_prefix.with_suffix(".xml")
        self.eegfile = f_prefix.with_suffix(".eeg")
        self.datfile = f_prefix.with_suffix(".dat")


class sessionname:
    def __init__(self, f_prefix):
        basePath = str(f_prefix.parent)
        self.sessionName = basePath.split("/")[-3] + basePath.split("/")[-2]
        self.name = basePath.split("/")[-3]
        self.day = basePath.split("/")[-2]
        self.basePath = Path(basePath)
        self.subname = f_prefix.stem

