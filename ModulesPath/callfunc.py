import os
from pathlib import Path

from parsePath import path2files

from lfpEvent import ripple, hswa
import numpy as np

from makeChanMap import recinfo
from MakePrmKlusta import makePrmPrb
from eventCorr import event_event
from artifactDetect import findartifact
from behavior import behavior_epochs
from sleepDetect import SleepScore
from MakePrmKlusta import makePrmPrb


class processData:
    # common parameters used frequently
    _lfpsRate = 1250

    def __init__(self, basePath):
        self.sessinfo = path2files(basePath)
        self.recinfo = recinfo(self.sessinfo)
        self.sessinfo.recinfo = self.recinfo

        self.epochs = behavior_epochs(self.sessinfo)
        self.makePrmPrb = makePrmPrb(self.sessinfo)
        self._trange = None

    @property
    def trange(self):
        return self._trange

    @trange.setter
    def trange(self, period):
        self._trange = period
        self.sessinfo.trange = period
        self.sessinfo.epochs = self.epochs
        # self.spksrt_param = makePrmPrb(sessinfo)
        self.ripple = ripple(self.sessinfo)
        self.swa = hswa(self.sessinfo)
        # self.artifact = findartifact(sessinfo)

        # for peristimuus histogram which needs ripple and swa
        self.sessinfo.swa = self.swa
        self.sessinfo.ripple = self.ripple
        self.eventpsth = event_event(self.sessinfo)

        self.brainstates = SleepScore(self.sessinfo)
