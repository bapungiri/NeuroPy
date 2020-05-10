import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import os
import linecache
import time
from datetime import datetime, timedelta
from pathlib import Path


def posPreprocess(basePath, posX, posY, video_starttime):

    for file in os.listdir(basePath):
        if file.endswith(".eeg"):
            subname = file[:-4]
            filePrefix = os.path.join(basePath, file[:-4])

    eegFileOG = Path(basePath, filePrefix + ".eeg")
    basics = np.load(filePrefix + "_basics.npy", allow_pickle=True)
    nChans = basics.item().get("nChans")
    sRate = basics.item().get("sRate")
    posX = np.asarray(posX)
    posY = np.asarray(posY)

    endFrames = len(np.memmap(eegFileOG, dtype="int16", mode="r")) / nChans

    if os.path.exists(Path(basePath, subname, "/og_files")):
        noisyFrames = np.load(basePath / "og_files" / "noisy_timestamps_fromOG.npy")
        nframes = np.load(basePath / "og_files" / "numframes_OG.npy")

    else:
        noisyFrames = []
        nframes = endFrames

    video_duration = len(posX) / (120 * 3600)
    video_endtime = video_starttime + timedelta(hours=video_duration)

    ephys_starttime = datetime.strptime(subname[-19:], "%Y-%m-%d_%H-%M-%S")

    time_diff = ephys_starttime - video_starttime

    ephys_duration = nframes / (3600 * 1250)
    ephys_endtime = ephys_starttime + timedelta(hours=ephys_duration)
    # time_record = np.arange(ephys_starttime, ephys_endtime, dtype="datetime64[h]")
    time_diff_end = video_endtime - ephys_endtime

    # t_ephys = np.arange(0, nframes) / 1250
    t_video = np.linspace(
        -time_diff.total_seconds(),
        nframes / 1250 + time_diff_end.total_seconds(),
        len(posX),
    )

    t_video_outside = np.argwhere((nframes / 1250 < t_video) | (t_video < 0))

    if not noisyFrames:
        ind_good = np.setdiff1d(np.arange(1, len(posX)), t_video_outside)
        print(len(ind_good))

    else:
        noisy_time = noisyFrames / 1250
        t_video_noisy = np.concatenate(
            [np.where(np.digitize(t_video, x) == 1)[0] for x in noisy_time], axis=0
        )

        t_video_noisy = np.union1d(t_video_outside, t_video_noisy)
        ind_good = np.setdiff1d(np.arange(1, len(posX)), t_video_noisy)

    t_video_keep = t_video[ind_good]
    posX_keep = posX[ind_good]
    posY_keep = posY[ind_good]
    time = np.arange(0, len(t_video_keep)) / 120

    return posX_keep, posY_keep, t_video_keep, time


def posfromFBX(fileName):
    fileName = str(fileName)

    xpos, ypos, zpos = [], [], []
    with open(fileName) as f:
        next(f)
        for i, line in enumerate(f):

            m = "".join(line)

            if "KeyCount" in m:
                track_begin = i + 2
                line_frame = linecache.getline(fileName, i + 2).strip().split(" ")
                total_frames = int(line_frame[1]) - 1
                break

    f.close()

    with open(fileName) as f:
        for _ in range(track_begin):
            next(f)

        for i, line in enumerate(f):
            # print(line)
            if len(xpos) > total_frames:
                break

            elif i < 1:
                print(i)
                line = line.strip()
                m = line.split(",")
                pos1 = m[1::5]
                print(pos1)

            else:
                line = line.strip()
                m = line.split(",")
                pos1 = m[2::5]

            xpos.extend(pos1)

        for _ in range(5):
            next(f)

        for i, line in enumerate(f):
            # print(line)
            if len(ypos) > total_frames:
                break

            elif i < 1:
                print(i)
                line = line.strip()
                m = line.split(",")
                pos1 = m[1::5]
                print(pos1)

            else:
                line = line.strip()
                m = line.split(",")
                pos1 = m[2::5]

            ypos.extend(pos1)

        for _ in range(5):
            next(f)

        for i, line in enumerate(f):
            # print(line)
            if len(zpos) > total_frames:
                break

            elif i < 1:
                print(i)
                line = line.strip()
                m = line.split(",")
                pos1 = m[1::5]
                print(pos1)

            else:
                line = line.strip()
                m = line.split(",")
                pos1 = m[2::5]

            # line = next(f)
            zpos.extend(pos1)

    f.close()
    xpos = [float(x) for x in xpos]
    ypos = [float(x) for x in ypos]
    zpos = [float(x) for x in zpos]

    return np.asarray(xpos), np.asarray(ypos), np.asarray(zpos)


def getStartTime(fileName):
    fileName = str(fileName)

    with open(fileName, newline="") as f:
        reader = csv.reader(f)
        row1 = next(reader)
        StartTime = [
            row1[i + 1] for i in range(len(row1)) if row1[i] == "Capture Start Time"
        ]
    # print(StartTime)
    tbegin = datetime.strptime(StartTime[0], "%Y-%m-%d %H.%M.%S.%f %p")
    return tbegin


def syncPos(posfile, ephysfile):
    posfile = str(posfile)
    ephysfile = str(ephysfile)

    ephysframes = np.load(ephysfile, mmap_mode="r")
    nframes = len(ephysframes)

    X, Y, Z = posfromFBX(posfile)
    vid_start = getStartTime(ephysfile[:-4] + ".csv")
    vid_dur = len(X) / (120 * 3600)
    vid_end = vid_start + timedelta(hours=vid_dur)

    ephys_start = datetime.strptime("2019-08-08_04-00-00 AM", "%Y-%m-%d_%H-%M-%S %p")
    ephys_dur = nframes / (3600 * 1250)
    ephys_end = ephys_start + timedelta(hours=ephys_dur)

    time_diff_start = ephys_start - vid_start
    time_diff_end = vid_end - ephys_end

    t_video = np.linspace(
        -time_diff_start.total_seconds(),
        nframes / 1250 + time_diff_end.total_seconds(),
        len(X),
    )

    t_video_outside = np.argwhere((nframes / 1250 < t_video) | (t_video < 0))

    ind_keep = np.setdiff1d(np.arange(1, len(X)), t_video_outside)

    return X[ind_keep], Y[ind_keep], Z[ind_keep]


class ExtractPosition:

    tracking_sRate = 120  # position sample rate

    def __init__(self, obj):
        """initiates position class

        Arguments:
            obj {class instance} -- should have the following attributes
                obj.sessinfo.files.position --> filename for storing the positions
        """
        self._obj = obj
        posfile = self._obj.sessinfo.files.position
        if os.path.exists(posfile):
            posInfo = self._load(posfile)
            self.x = posInfo.item().get("X")  # in seconds
            self.y = posInfo.item().get("Y")  # in seconds
            self.t = posInfo.item().get("time")  # in seconds

        else:
            "Position file does not exist....did not load _position.npy"

    def _load(self, posfile):
        return np.load(posfile, allow_pickle=True)

    def getPosition(self):
        basePath = Path(self._obj.sessinfo.basePath)
        posFolder = basePath / "position"
        for file in os.listdir(posFolder):

            if file.endswith(".csv"):
                print(file)
                fileName = Path(posFolder, file)

                with open(fileName, newline="") as f:
                    reader = csv.reader(f)
                    row1 = next(reader)
                    StartTime = [
                        row1[i + 1]
                        for i in range(len(row1))
                        if row1[i] == "Capture Start Time"
                    ]
                # print(StartTime)
                tbegin = datetime.strptime(StartTime[0], "%Y-%m-%d %H.%M.%S.%f %p")

            elif file.endswith(".fbx"):
                print(file)
                xpos, ypos, zpos = posfromFBX(Path(posFolder, file))

                # self.posX = xpos
                # self.posY = zpos
                # # self.posZ = zpos
                # self.frames = np.arange(1, len(xpos) + 1)
        posX, posY, time_orig, time = posPreprocess(basePath, xpos, zpos, tbegin)

        posVar = {
            "X": posX,
            "Y": posY,
            "time_orig": time_orig,
            "time": time,
            "pos_sRate": self.tracking_sRate,
        }

        np.save(self._obj.sessinfo.files.position, posVar)

    def plot(self):

        plt.clf()
        plt.plot(self.x, self.y)

    def Speed(self):
        pass

    def export2Neuroscope(self):

        # neuroscope only displays positive values so translating the coordinates
        x = self.x + abs(min(self.x))
        y = self.y + abs(min(self.y))
        print(max(x))
        print(max(y))

        filename = self._obj.sessinfo.files.filePrefix.with_suffix(".pos")
        with filename.open("w") as f:
            for xpos, ypos in zip(x, y):
                f.write(f"{xpos} {ypos}\n")

    def _posPreprocess(self, posX, posY, tbegin):

        eegFileOG = self._obj.sessinfo.recfiles.eegfile
        nChans = self._obj.recinfo.nChans
        posX = np.asarray(posX)
        posY = np.asarray(posY)

        endFrames = len(np.memmap(eegFileOG, dtype="int16", mode="r")) / nChans

        if os.path.exists(Path(basePath, subname, "/og_files")):
            noisyFrames = np.load(basePath / "og_files" / "noisy_timestamps_fromOG.npy")
            nframes = np.load(basePath / "og_files" / "numframes_OG.npy")

        else:
            noisyFrames = []
            nframes = endFrames

        video_duration = len(posX) / (120 * 3600)
        video_endtime = video_starttime + timedelta(hours=video_duration)

        ephys_starttime = datetime.strptime(subname[-19:], "%Y-%m-%d_%H-%M-%S")

        time_diff = ephys_starttime - video_starttime

        ephys_duration = nframes / (3600 * 1250)
        ephys_endtime = ephys_starttime + timedelta(hours=ephys_duration)
        # time_record = np.arange(ephys_starttime, ephys_endtime, dtype="datetime64[h]")
        time_diff_end = video_endtime - ephys_endtime

        # t_ephys = np.arange(0, nframes) / 1250
        t_video = np.linspace(
            -time_diff.total_seconds(),
            nframes / 1250 + time_diff_end.total_seconds(),
            len(posX),
        )

        t_video_outside = np.argwhere((nframes / 1250 < t_video) | (t_video < 0))

        if not noisyFrames:
            ind_good = np.setdiff1d(np.arange(1, len(posX)), t_video_outside)
            print(len(ind_good))

        else:
            noisy_time = noisyFrames / 1250
            t_video_noisy = np.concatenate(
                [np.where(np.digitize(t_video, x) == 1)[0] for x in noisy_time], axis=0
            )

            t_video_noisy = np.union1d(t_video_outside, t_video_noisy)
            ind_good = np.setdiff1d(np.arange(1, len(posX)), t_video_noisy)

        t_video_keep = t_video[ind_good]
        posX_keep = posX[ind_good]
        posY_keep = posY[ind_good]
        time = np.arange(0, len(t_video_keep)) / 120

        return posX_keep, posY_keep, t_video_keep, time
