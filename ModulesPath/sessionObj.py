import os
from pathlib import Path

import numpy as np

from eventCorr import hswa_ripple
from lfpEvent import hswa, swr
from makeChanMap import ExtractChanXml
from MakePrmKlusta import makePrmPrb


class session(ExtractChanXml, makePrmPrb, swr, hswa, hswa_ripple):
    def __init__(self, basePath):
        super().__init__(basePath)
