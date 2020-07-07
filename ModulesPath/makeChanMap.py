import os
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path


class recinfo:
    nShanks = 8

    def __init__(self, obj):

        self._obj = obj

        if Path(self._obj.sessinfo.files.basics).is_file():
            self._intialize()

    def _intialize(self):

        myinfo = np.load(self._obj.sessinfo.files.basics, allow_pickle=True).item()
        self.sampfreq = myinfo["sRate"]
        self.channels = myinfo["channels"]
        self.nChans = myinfo["nChans"]
        self.lfpSrate = myinfo["lfpSrate"]
        self.channelgroups = myinfo["channelgroups"]
        self.badchans = myinfo["badchans"]
        self.goodchans = np.setdiff1d(self.channels, self.badchans, assume_unique=True)

    def makerecinfo(self, badchans=None):

        if badchans is None and (self._obj.sessinfo.files.badchans).is_file():
            badchans = np.load(self._obj.sessinfo.files.badchans)

        myroot = ET.parse(self._obj.sessinfo.recfiles.xmlfile).getroot()

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
            "subname": self._obj.sessinfo.session.subname,
            "sessionName": self._obj.sessinfo.session.sessionName,
            "lfpSrate": 1250,
            "badchans": badchans,
        }

        np.save(self._obj.sessinfo.files.basics, basics)
        print(f"_basics.npy created for {self._obj.sessinfo.session.sessionName}")

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
        datfile = self._obj.sessinfo.recfiles.datfile
        data = np.memmap(datfile, dtype="int16", mode="r")

        return len(data) / nChans
