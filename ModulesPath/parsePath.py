import numpy as np
import os
from pathlib import Path


class name2path:
    def __init__(self, basePath):
        self.sessionName = basePath.split("/")[-3] + basePath.split("/")[-2]
        self.name = basePath.split("/")[-3]
        self.day = basePath.split("/")[-2]
        self.basePath = basePath
        for file in os.listdir(basePath):
            if file.endswith(".xml"):
                self.subname = file[:-4]
                self.filename = os.path.join(basePath, file)
                self.filePrefix = os.path.join(basePath, file[:-4])
