import os
from pathlib import Path

from lfpEvent import ripple
import numpy as np

# from parsePath import path2files
from makeChanMap import recinfo

from eventCorr import event_event


class processData:
    # common parameters used frequently
    __lfpsRate = 1250

    def __init__(self, basePath):

        self.recinfo = recinfo(basePath)
        self.ripple = ripple(basePath)
        self.eventpsth = event_event(basePath)
