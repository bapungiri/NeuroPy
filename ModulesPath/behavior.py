import os
import numpy as np

# from parsePath import path2files


class behavior_epochs:
    nShanks = 8

    def __init__(self, obj):
        myinfo = obj
        recinfo = np.load(myinfo.sessinfo.files.epochs, allow_pickle=True).item()
        # print(recinfo.keys())
        self.pre = recinfo["PRE"]
        self.maze = recinfo["MAZE"]
        self.post = recinfo["POST"]

    def makebehavior(self):
        pass
