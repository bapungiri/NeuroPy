import os
from pathlib import Path

from parsePath import path2files

from lfpEvent import ripple, hswa
import numpy as np

from makeChanMap import recinfo
from MakePrmKlusta import makePrmPrb
from eventCorr import event_event
from artifactDetect import findartifact
from getPosition import ExtractPosition
from behavior import behavior_epochs
from sleepDetect import SleepScore
from MakePrmKlusta import makePrmPrb
from getSpikes import spikes
from replay import Replay
from pfPlot import pf
from decoders import DecodeBehav


class processData:
    # common parameters used frequently
    _lfpsRate = 1250

    def __init__(self, basePath):
        self.sessinfo = path2files(basePath)
        self.recinfo = recinfo(self)

        self.position = ExtractPosition(self)
        self.epochs = behavior_epochs(self)
        self.artifact = findartifact(self)
        self.makePrmPrb = makePrmPrb(self)
        self._trange = None

    @property
    def trange(self):
        return self._trange

    @trange.setter
    def trange(self, period):
        self._trange = period
        self.spikes = spikes(self)
        self.brainstates = SleepScore(self)
        self.ripple = ripple(self)
        self.swa = hswa(self)
        self.eventpsth = event_event(self)
        self.placefield = pf(self)
        self.replay = Replay(self)
        self.decode = DecodeBehav(self)
