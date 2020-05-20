import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import os
import linecache
import time
from datetime import datetime, timedelta
from pathlib import Path


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

    with open(fileName) as f:
        for _ in range(track_begin):
            next(f)

        for i, line in enumerate(f):
            # print(line)
            if len(xpos) > total_frames:
                break

            elif i < 1:
                # print(i)
                line = line.strip()
                m = line.split(",")
                pos1 = m[1::5]
                # print(pos1)

            else:
                line = line.strip()
                m = line.split(",")
                pos1 = m[2::5]

            xpos.extend(pos1)

        for line in f:

            if "KeyCount" in line:
                break
            else:
                next(f)

        pos1 = []
        for i, line in enumerate(f):
            # print(line)
            if len(ypos) > total_frames:
                break

            elif i < 1:
                # print(i)
                line = line.strip()
                m = line.split(",")
                pos1 = m[1::5]
                # print(pos1)

            else:
                line = line.strip()
                m = line.split(",")
                pos1 = m[2::5]

            ypos.extend(pos1)

        for line in f:

            if "KeyCount" in line:
                break
            else:
                next(f)

        pos1 = []

        for i, line in enumerate(f):
            # print(line)

            if len(zpos) > total_frames:
                break

            elif i < 1:
                # print(i)
                line = line.strip()
                m = line.split(",")
                pos1 = m[1::5]
                # print(pos1)

            else:
                line = line.strip()
                m = line.split(",")
                pos1 = m[2::5]

            # line = next(f)
            zpos.extend(pos1)

    xpos = [float(_) for _ in xpos]
    ypos = [float(_) for _ in ypos]
    zpos = [float(_) for _ in zpos]

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
            posInfo = self._load(posfile).item()
            self.x = posInfo["x"]  # in seconds
            self.y = posInfo["y"]  # in seconds
            self.t = posInfo["time"]  # in seconds
            self.datetime = posInfo["datetime"]  # in seconds

        else:
            "Position file does not exist....did not load _position.npy"

    def _load(self, posfile):
        return np.load(posfile, allow_pickle=True)

    def getPosition(self):
        sRate = self._obj.recinfo.sampfreq  # .dat file sampling frequency
        basePath = Path(self._obj.sessinfo.basePath)
        metadata = self._obj.sessinfo.metadata

        nfiles = metadata.count()["StartTime"]

        # collecting timepoints related to .dat file
        data_time = []
        for i, file_time in enumerate(metadata["StartTime"][:nfiles]):
            tbegin = datetime.strptime(file_time, "%Y-%m-%d_%H-%M-%S")
            nframes = metadata["nFrames"][i]
            duration = pd.Timedelta(nframes / sRate, unit="sec")
            tend = tbegin + duration
            trange = pd.date_range(
                start=tbegin,
                end=tend,
                periods=int(duration.seconds * self.tracking_sRate),
            )
            data_time.extend(trange)

        data_time = pd.to_datetime(data_time)

        # deleting intervals that were deleted from .dat file after concatenating
        ndeletedintervals = metadata.count()["deletedStart (minutes)"]
        for i in range(ndeletedintervals):
            tnoisy_begin = data_time[0] + pd.Timedelta(
                metadata["deletedStart (minutes)"][i], unit="m"
            )
            tnoisy_end = data_time[0] + pd.Timedelta(
                metadata["deletedEnd (minutes)"][i], unit="m"
            )

            del_index = np.where((data_time > tnoisy_begin) & (data_time < tnoisy_end))[
                0
            ]

            data_time = np.delete(data_time, del_index)
            # data_time = data_time.indexer_between_time(
            #     pd.Timestamp(tnoisy_end), pd.Timestamp(tnoisy_begin)
            # )

        # ===== collecting timepoints related to position tracking =====
        posFolder = basePath / "position"
        posfiles = sorted(posFolder.glob("*.csv"))
        postime, posx, posy, posz = [], [], [], []
        for file in posfiles:
            print(file)

            fileinfo = pd.read_csv(file, header=None, nrows=1)
            # required values are in column 11 and 13 of .csv file
            tbegin = datetime.strptime(fileinfo.iloc[0][11], "%Y-%m-%d %H.%M.%S.%f %p")
            nframes = fileinfo.iloc[0][13]
            duration = pd.Timedelta(nframes / self.tracking_sRate, unit="sec")
            tend = tbegin + duration
            trange = pd.date_range(start=tbegin, end=tend, periods=nframes)

            x, y, z = posfromFBX(file.with_suffix(".fbx"))

            postime.extend(trange)
            posx.extend(x)
            posy.extend(y)
            posz.extend(z)

        postime = pd.to_datetime(postime[: len(posx)])
        posx = np.asarray(posx)
        posy = np.asarray(posy)
        posz = np.asarray(posz)

        # ======= interpolating positions for recorded data======
        xdata = np.interp(data_time, postime, posx)
        ydata = np.interp(data_time, postime, posy)
        zdata = np.interp(data_time, postime, posz)
        time = np.linspace(0, len(xdata) / 120, len(xdata))

        posVar = {
            "x": xdata,
            "y": zdata,  # as in optitrack the z coordinates gives the y information
            "time": time,
            "datetime": data_time,
            "trackingsRate": self.tracking_sRate,
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
