import os
import tkinter as tk
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd


class Recinfo:
    nShanks = 8

    def __init__(self, basePath):
        self.basePath = Path(basePath)

        for file in os.listdir(basePath):
            if file.endswith(".xml"):
                xmlfile = self.basePath / file
                filePrefix = xmlfile.with_suffix("")

        self.session = sessionname(filePrefix)
        self.files = files(filePrefix)
        self.recfiles = recfiles(filePrefix)
        # self.loadfile = loadfile(filePrefix)

        if Path(self.files.basics).is_file():
            self._intialize()

    def _intialize(self):

        myinfo = np.load(self.files.basics, allow_pickle=True).item()
        self.sampfreq = myinfo["sRate"]
        self.channels = myinfo["channels"]
        self.nChans = myinfo["nChans"]
        self.lfpSrate = myinfo["lfpSrate"]
        self.channelgroups = myinfo["channelgroups"]
        self.badchans = myinfo["badchans"]
        self.goodchans = np.setdiff1d(self.channels, self.badchans, assume_unique=True)
        self.goodchangrp = [
            list(np.setdiff1d(_, self.badchans).astype(int)) for _ in self.channelgroups
        ][: self.nShanks]

    @property
    def metadata(self):
        metadatafile = Path(str(self.files.filePrefix) + "_metadata.csv")
        if metadatafile.is_file():
            metadata = pd.read_csv(metadatafile)

        else:
            val = input("Do you want to create metadata, Yes or No: ")
            if val in ["Y", "y", "yes", "Yes", "YES"]:

                def show_entry_fields():
                    print("First Name: %s\nLast Name: %s" % (e1.get(), e2.get()))

                master = tk.Tk()
                tk.Label(master, text="First Name").grid(row=0)
                tk.Label(master, text="Last Name").grid(row=1)

                e1 = tk.Entry(master)
                e2 = tk.Entry(master)

                e1.grid(row=0, column=1)
                e2.grid(row=1, column=1)

                tk.Button(master, text="Quit", command=master.quit).grid(
                    row=3, column=0, sticky=tk.W, pady=4
                )
                tk.Button(master, text="Show", command=show_entry_fields).grid(
                    row=3, column=1, sticky=tk.W, pady=4
                )

                tk.mainloop()

        return metadata

    def makerecinfo(self, badchans=None):

        if badchans is None and (self.files.badchans).is_file():
            badchans = np.load(self.files.badchans)

        myroot = ET.parse(self.recfiles.xmlfile).getroot()

        chan_session = []
        channelgroups = []
        for x in myroot.findall("anatomicalDescription"):
            for y in x.findall("channelGroups"):
                for z in y.findall("group"):
                    chan_group = []
                    for chan in z.findall("channel"):
                        chan_session.append(int(chan.text))

                        chan_group.append(int(chan.text))
                    channelgroups.append(chan_group)

        for sf in myroot.findall("acquisitionSystem"):
            sampfreq = int(sf.find("samplingRate").text)

        nChans = len(chan_session)

        basics = {
            "sRate": sampfreq,
            "channels": chan_session,
            "nChans": nChans,
            "channelgroups": channelgroups,
            "nShanks": self.nShanks,
            "subname": self.session.subname,
            "sessionName": self.session.sessionName,
            "lfpSrate": 1250,
            "badchans": badchans,
        }

        np.save(self.files.basics, basics)
        print(f"_basics.npy created for {self.session.sessionName}")

        # laods attributes in runtime so doesn't lead reloading of entire class instance
        self._intialize()

    def probemap(self, probetype="diagbio"):
        changroup = self.channelgroups
        nShanks = self.nShanks

        if len(changroup[0]) == 16:
            probetype = "diagbio"
        if len(changroup[0]) == 8:
            probetype = "buzsaki"

        changroup = changroup[:nShanks]
        if probetype == "diagbio":

            xcoord, ycoord = [], []
            for i in range(nShanks):
                xpos = [10 * (_ % 2) + i * 150 for _ in range(16)]
                ypos = [15 * 16 - _ * 15 for _ in range(16)]
                xcoord.extend(xpos)
                ycoord.extend(ypos)

        if probetype == "buzsaki":

            xp = [0, 37, 4, 33, 8, 29, 12, 20]
            yp = np.arange(160, 0, -20)
            xcoord, ycoord = [], []
            for i in range(nShanks):
                xpos = [xp[_] + i * 200 for _ in range(8)]
                ypos = [yp[_] for _ in range(8)]
                xcoord.extend(xpos)
                ycoord.extend(ypos)

        return [xcoord, ycoord]

    @property
    def getNframesDat(self):
        nChans = self.nChans
        datfile = self.recfiles.datfile
        data = np.memmap(datfile, dtype="int16", mode="r")

        return len(data) / nChans

    def geteeg(self, chans, timeRange=None, frames=None):
        """Returns eeg signal for given channels and timeperiod or selected frames

        Args:
            chans (list): list of channels required index should in order of binary file
            timeRange (list, optional): In seconds and must have length 2.
            frames (list, optional): Required frames from the eeg data.

        Returns:
            eeg: [array of channels x timepoints]
        """
        eegfile = self.recfiles.eegfile
        eegSrate = self.lfpSrate
        nChans = self.nChans

        eeg = np.memmap(eegfile, dtype="int16", mode="r")
        eeg = np.memmap.reshape(eeg, (nChans, len(eeg) // nChans), order="F")

        if timeRange is not None:
            assert len(timeRange) == 2
            frameStart = int(timeRange[0] * eegSrate)
            frameEnd = int(timeRange[1] * eegSrate)
            eeg = eeg[:, frameStart:frameEnd]
        elif frames is not None:
            eeg = eeg[:, frames]

        eeg = eeg[chans, :]
        return eeg

    def plotChanPos(self, chans=None, ax=None, colors=None):

        nShanks = self._obj.nShanks
        channelgrp = self._obj.channelgroups[:nShanks]
        lfpchans = np.asarray([chan for shank in channelgrp for chan in shank])

        chans2plot = chans
        chan_rank = np.where(np.isin(lfpchans, chans2plot))[0]
        xpos, ypos = self._obj.recinfo.probemap()
        xpos = np.asarray(xpos)
        ypos = np.asarray(ypos)

        if ax is None:
            fig = plt.figure(1, figsize=(10, 15))
            gs = gridspec.GridSpec(1, 1, figure=fig)
            fig.subplots_adjust(hspace=0.3)
            ax = fig.add_subplot(gs[0])

        ax.scatter(xpos, ypos, s=4, color="gray", zorder=1)
        if colors is None:
            ax.scatter(xpos[chan_rank], ypos[chan_rank], c="red", s=20, zorder=2)
        else:
            ax.scatter(xpos[chan_rank], ypos[chan_rank], c=colors, s=40, zorder=2)


class files:
    def __init__(self, f_prefix):
        self.filePrefix = f_prefix

        self.spikes = Path(str(f_prefix) + "_spikes.npy")

        self.basics = Path(str(f_prefix) + "_basics.npy")
        self.badchans = Path(str(f_prefix) + "_badChans.npy")
        self.position = Path(str(f_prefix) + "_position.npy")
        self.epochs = Path(str(f_prefix) + "_epochs.npy")
        self.ripplelfp = Path(str(f_prefix) + "_BestRippleChans.npy")
        self.ripple_evt = Path(str(f_prefix) + "_ripples.npy")
        self.spindle_evt = Path(str(f_prefix) + "_spindles.npy")
        self.spindlelfp = Path(str(f_prefix) + "_BestSpindleChan.npy")

        self.thetalfp = Path(str(f_prefix) + "_BestThetaChan.npy")
        self.theta_evt = Path(str(f_prefix) + "_thetaevents.npy")
        self.sessionepoch = Path(str(f_prefix) + "_epochs.npy")
        self.hwsa_ripple = Path(str(f_prefix) + "_hswa_ripple.npy")
        self.sws_states = Path(str(f_prefix) + "_sws.npy")
        self.slow_wave = Path(str(f_prefix) + "_hswa.npy")
        self.corr_emg = Path(str(f_prefix) + "_emg.npy")
        self.spectrogram = Path(str(f_prefix) + "_sxx.npy")
        self.stateparams = Path(str(f_prefix) + "_stateparams.pkl")
        self.states = Path(str(f_prefix) + "_states.pkl")


# TODO auto file loading functionality
class loadfile:
    def __init__(self, filename):
        self.name = filename

    def load(self):

        if self.name.suffix == ".pkl":
            pd.read_pickle(self.name)


class recfiles:
    def __init__(self, f_prefix):

        self.xmlfile = f_prefix.with_suffix(".xml")
        self.eegfile = f_prefix.with_suffix(".eeg")
        self.datfile = f_prefix.with_suffix(".dat")


class sessionname:
    def __init__(self, f_prefix):
        basePath = str(f_prefix.parent)
        self.sessionName = basePath.split("/")[-2] + basePath.split("/")[-1]
        self.name = basePath.split("/")[-2]
        self.day = basePath.split("/")[-1]
        # self.basePath = Path(basePath)
        self.subname = f_prefix.stem
