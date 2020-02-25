import numpy as np
import os
from pathlib import Path


class name2path:
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
