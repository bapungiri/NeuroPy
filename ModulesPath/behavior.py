import os
import numpy as np
from parsePath import path2files


class behavior_epochs(path2files):
    nShanks = 8

    def __init__(self, basePath):
        super().__init__(basePath)

        recinfo = np.load(self._files.epochs, allow_pickle=True).item()
        # print(recinfo.keys())
        self.pre = recinfo["PRE"]
        self.maze = recinfo["MAZE"]
        self.post = recinfo["POST"]

    def makebehavior(self):
        pass

