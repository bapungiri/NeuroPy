import os
from pathlib import Path

# from parsePath import path2files

from lfpEvent import ripple
import numpy as np

from makeChanMap import recinfo
from MakePrmKlusta import makePrmPrb
from eventCorr import event_event
from artifactDetect import findartifact


class processData:
    # common parameters used frequently
    __lfpsRate = 1250

    def __init__(self, basePath):

        self.spksrt_param = makePrmPrb(basePath)
        self.recinfo = recinfo(basePath)
        self.ripple = ripple(basePath)
        self.eventpsth = event_event(basePath)
        self.artifact = findartifact(basePath)
