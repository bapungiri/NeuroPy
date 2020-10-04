import os
from pathlib import Path

import numpy as np
from pandas.core.dtypes import base

from artifactDetect import findartifact
from behavior import behavior_epochs
from decoders import DecodeBehav
from eventCorr import event_event
from getPosition import ExtractPosition
from getSpikes import spikes
from lfpEvent import Hswa, Ripple, Spindle, Theta

# from makeChanMap import recinfo
from MakePrmKlusta import makePrmPrb
from parsePath import Recinfo
from pfPlot import pf
from replay import Replay
from sessionUtil import SessionUtil
from sleepDetect import SleepScore
from spkEvent import PBE, LocalSleep
from viewerData import SessView


class processData:
    # common parameters used frequently
    lfpsRate = 1250

    def __init__(self, basepath):
        # self.sessinfo = path2files(basePath)
        self.recinfo = Recinfo(basepath)

        self.position = ExtractPosition(self.recinfo)
        self.epochs = behavior_epochs(self.recinfo)
        self.artifact = findartifact(self.recinfo)
        self.makePrmPrb = makePrmPrb(self.recinfo)
        self.utils = SessionUtil(self.recinfo)

        self.spikes = spikes(self.recinfo)
        self.theta = Theta(self.recinfo)
        self.brainstates = SleepScore(self.recinfo)
        self.ripple = Ripple(self.recinfo)
        self.swa = Hswa(self.recinfo)
        self.spindle = Spindle(self.recinfo)
        # self.eventpsth = event_event(self.recording)
        self.placefield = pf(self.recinfo)
        self.replay = Replay(self.recinfo)
        self.decode = DecodeBehav(self.recinfo)
        self.localsleep = LocalSleep(self.recinfo)
        self.viewdata = SessView(self.recinfo)
        self.pbe = PBE(self.recinfo)
