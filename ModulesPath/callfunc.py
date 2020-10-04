import os
from pathlib import Path

import numpy as np

from artifactDetect import findartifact
from behavior import behavior_epochs
from decoders import DecodeBehav
from eventCorr import event_event
from getPosition import ExtractPosition
from getSpikes import spikes
from lfpEvent import Hswa, Ripple, Spindle, Theta
from makeChanMap import recinfo
from MakePrmKlusta import makePrmPrb
from parsePath import path2files
from pfPlot import pf
from replay import Replay
from sessionUtil import SessionUtil
from sleepDetect import SleepScore
from spkEvent import PBE, LocalSleep
from viewerData import SessView


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
        self.utils = SessionUtil(self)
        self._trange = None

    @property
    def trange(self):
        return self._trange

    @trange.setter
    def trange(self, period):
        self._trange = period
        self.spikes = spikes(self)
        self.theta = Theta(self)
        self.brainstates = SleepScore(self)
        self.ripple = Ripple(self)
        self.swa = Hswa(self)
        self.spindle = Spindle(self)
        self.eventpsth = event_event(self)
        self.placefield = pf(self)
        self.replay = Replay(self)
        self.decode = DecodeBehav(self)
        self.localsleep = LocalSleep(self)
        self.viewdata = SessView(self)
        self.pbe = PBE(self)
