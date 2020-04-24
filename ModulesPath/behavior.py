# import os
import numpy as np
from pathlib import Path

# from parsePath import path2files


class behavior_epochs:
    nShanks = 8

    def __init__(self, obj):
        self._obj = obj

        if Path(self._obj.sessinfo.files.epochs).is_file():
            epochs = np.load(self._obj.sessinfo.files.epochs, allow_pickle=True).item()
            self.pre = epochs["PRE"]
            self.maze = epochs["MAZE"]
            self.post = epochs["POST"]
            self.totalduration = (
                np.diff(self.pre) + np.diff(self.maze) + np.diff(self.post)
            )[0]

        else:
            print("Epochs file does not exist...did not load epochs")

    def makebehavior(self):
        pass
