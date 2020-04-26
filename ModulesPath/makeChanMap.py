import os
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path


class recinfo:
    nShanks = 8

    def __init__(self, obj):

        self._obj = obj
        if Path(self._obj.sessinfo.files.epochs).is_file():

            myinfo = np.load(self._obj.sessinfo.files.basics, allow_pickle=True).item()
            self.sampfreq = myinfo["sRate"]
            self.channels = myinfo["channels"]
            self.nChans = myinfo["nChans"]
            self.lfpSrate = 1250
            self.channelgroups = myinfo["channelgroups"]
            self.badchans = myinfo["badchans"]

    def makerecinfo(self, badchans=None):

        myroot = ET.parse(self._obj.sessinfo.recfiles.xmlfile).getroot()

        self.chan_session = []
        self.channelgroups = []
        for x in myroot.findall("anatomicalDescription"):
            for y in x.findall("channelGroups"):
                for z in y.findall("group"):
                    chan_group = []
                    for chan in z.findall("channel"):
                        self.chan_session.append(int(chan.text))
                        chan_group.append(int(chan.text))
                    self.channelgroups.append(chan_group)

        for sf in myroot.findall("acquisitionSystem"):
            self.sampfreq = int(sf.find("samplingRate").text)

        self.nChans = len(self.chan_session)

        basics = {
            "sRate": self.sampfreq,
            "channels": self.chan_session,
            "nChans": self.nChans,
            "channelgroups": self.channelgroups,
            "nShanks": self.nShanks,
            "subname": self._obj.sessinfo.session.subname,
            "sessionName": self._obj.sessinfo.session.sessionName,
            "lfpSrate": 1250,
            "badchans": badchans,
        }

        np.save(self._obj.sessinfo.files.basics, basics)
        print(f"_basics.npy created for {self._obj.sessinfo.session.sessionName}")

    def probemap(self, probetype="diagbio"):
        changroup = self.channelgroups
        nShanks = self.nShanks

        changroup = changroup[:nShanks]
        if probetype == "diagbio":

            xcoord, ycoord = [], []
            for i in range(nShanks):
                xpos = [10 * (_ % 2) + i * 30 for _ in range(16)]
                ypos = [15 * 16 - _ * 15 for _ in range(16)]
                xcoord.extend(xpos)
                ycoord.extend(ypos)

        return [xcoord, ycoord]
