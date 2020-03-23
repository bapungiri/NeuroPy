import os
from pathlib import Path

# from parsePath import path2files

from lfpEvent import ripple
import numpy as np

from makeChanMap import recinfo
from MakePrmKlusta import makePrmPrb
from eventCorr import event_event
from artifactDetect import findartifact
from behavior import behavior_epochs


class processData:
    # common parameters used frequently
    _lfpsRate = 1250

    def __init__(self, basePath):
        self.recinfo = recinfo(basePath)
        self.epochs = behavior_epochs(basePath)
        self._trange = None

    @property
    def trange(self):
        return self._trange

    @trange.setter
    def trange(self, period):
        self._trange = period
        self.spksrt_param = makePrmPrb(basePath)
        self.ripple = ripple(basePath)
        self.eventpsth = event_event(basePath)
        self.artifact = findartifact(basePath)
