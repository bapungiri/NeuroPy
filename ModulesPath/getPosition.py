import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import os
import linecache
import time
from datetime import datetime


class ExtractPosition:

    optitrack_sRate = 120  # position sample rate

    def __init__(self, basePath):
        # self.sessionName = os.path.basename(os.path.normpath(basePath))
        self.sessionName = basePath.split("/")[-2] + basePath.split("/")[-1]
        self.basePath = basePath

        for file in os.listdir(basePath):
            if file.endswith(".eeg"):

                self.subname = file[:-4]
                self.filename = os.path.join(basePath, file)
                self.filePrefix = os.path.join(basePath, file[:-4])

        # checking if position file already exists
        if os.path.exists(self.basePath + "_position.npy"):
            posInfo = np.load(self.basePath + "_positon.npy", allow_pickle=True)
            self.posX = posInfo.item().get("X")  # in seconds
            self.posY = posInfo.item().get("Y")  # in seconds
            self.frames = posInfo.item().get("frames")  # in seconds
            self.tbegin = posInfo.item().get("begin")

        # Run to generate new position file
        else:
            posFolder = basePath + "position/"
            for file in os.listdir(posFolder):

                if file.endswith(".csv"):
                    print(file)
                    fileName = posFolder + file

                    with open(fileName, newline="") as f:
                        reader = csv.reader(f)
                        row1 = next(reader)
                        StartTime = [
                            row1[i + 1]
                            for i in range(len(row1))
                            if row1[i] == "Capture Start Time"
                        ]
                    # print(StartTime)
                    self.tbegin = datetime.strptime(
                        StartTime[0], "%Y-%m-%d %H.%M.%S.%f %p"
                    )
                    # print(tbegin)

                    # positionStruct = pd.read_csv(fileName, header=5)
                    # positionStruct = positionStruct.iloc[:, [0, 1, 6, 7, 8]]
                    # positionStruct.interpolate(axis=0)

                    # self.frames = positionStruct.iloc[:, 0]
                    # self.time = positionStruct.iloc[:, 1]
                    # self.posX = positionStruct.iloc[:, 2]
                    # self.posY = positionStruct.iloc[:, 3]
                    # self.posZ = positionStruct.iloc[:, 4]
                    # self.dt = self.time[1] - self.time[0]

                    # # self.time = self.time + 5.457 + 10041.21
                    # # posVar = {}
                    # # posVar["X"] = self.posX
                    # # posVar["Y"] = self.posZ
                    # # posVar["time"] = self.time

                    # np.save(basePath + self.subname + "_position.npy", posVar)

                elif file.endswith(".fbx"):
                    print(file)
                    fileName = posFolder + file

                    # k = 830

                    with open(fileName) as f:
                        next(f)
                        for i, line in enumerate(f):

                            m = "".join(line)

                            if "RawSegs" in m:
                                track_begin = i + 3
                                line_frame = (
                                    linecache.getline(fileName, i + 3)
                                    .strip()
                                    .split(",")
                                )
                                total_frames = float(line_frame[3])
                                break

                    f.close()
                    xpos, ypos, zpos = [], [], []
                    with open(fileName) as f:
                        for i in range(track_begin):
                            next(f)

                        for i, line in enumerate(f):

                            if i > track_begin:
                                line = line.strip()
                                m = line.split(",")

                                xpos.append(m[1])
                                ypos.append(m[2])
                                zpos.append(m[3])

                            if i == track_begin + total_frames - 1:
                                break

                        self.posX = list(map(float, xpos))
                        self.posY = list(map(float, ypos))
                        self.posZ = list(map(float, zpos))
                        self.frames = np.arange(1, len(self.posX) + 1)
                        self.time = np.arange(1, len(self.posX)) * (1 / 120)
                    f.close()

            posVar = {}
            posVar["X"] = self.posX
            posVar["Y"] = self.posZ
            posVar["frames"] = self.frames
            posVar["begin"] = self.tbegin
            posVar["pos_sRate"] = self.optitrack_sRate
            np.save(basePath + self.subname + "_position.npy", posVar)

    def plotPosition(self):

        plt.clf()
        plt.plot(self.posX, self.posZ)

    def getMazeFrames(self):

        # plt.clf()

        # pts = plt.ginput(2, timeout=-1)
        def tellme(s):
            print(s)
            plt.title(s, fontsize=16)
            plt.draw()

        # Copy to clipboard

        # Define a rectangle by clicking two points

        plt.clf()
        plt.setp(plt.gca(), autoscale_on=True)
        plt.plot(self.frames[::4], self.posZ[::4])

        tellme("You will define a rectangle for track, click to begin")

        plt.waitforbuttonpress()

        while True:
            pts = []
            while len(pts) < 2:
                tellme("Select 2 edges with mouse")
                pts = np.asarray(plt.ginput(2, timeout=-1))
                if len(pts) < 2:
                    tellme("Too few points, starting over")
                    time.sleep(1)  # Wait a second

                pts = np.asarray(
                    [[pts[0, 0], 400], [pts[0, 0], 0], [pts[1, 0], 0], [pts[1, 0], 400]]
                )

            ph = plt.fill(pts[:, 0], pts[:, 1], "r", lw=2, alpha=0.6)

            tellme("Happy? Key click for yes, mouse click for no")

            if plt.waitforbuttonpress():
                break

            # Get rid of fill
            for p in ph:
                p.remove()
        self.corrds = pts
        self.maze_start = (int(pts[0][0])) / self.optitrack_sRate  # in seconds
        self.maze_end = int(pts[2][0]) / self.optitrack_sRate  # in seconds

        pre_time = np.array([0, self.maze_start - 1])
        maze_time = np.array([self.maze_start, self.maze_end])
        post_time = np.array(
            [self.maze_end + 1, len(self.frames) / self.optitrack_sRate]
        )
        epoch_times = {"PRE": pre_time, "MAZE": maze_time, "POST": post_time}

        np.save(self.basePath + self.subname + "_epochs.npy", epoch_times)

    def Speed(self):
        location = np.sqrt((self.posZ) ** 2 + (self.posX) ** 2)
        spd = np.abs(np.diff(location)) / self.dt

        self.speed = spd.tolist()
        return self.speed
