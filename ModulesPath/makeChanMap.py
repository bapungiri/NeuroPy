import os
import numpy as np
import xml.etree.ElementTree as ET


class recinfo:
    nShanks = 8

    def __init__(self, obj):

        self._obj = obj

        myinfo = np.load(self._obj.files.basics, allow_pickle=True).item()
        badchans = np.load(self._obj.files.badchans)
        # print(recinfo.keys())
        self.sampfreq = myinfo["sRate"]
        self.channels = myinfo["channels"]
        self.nChans = myinfo["nChans"]
        self.lfpSrate = 1250
        self.channelgroups = myinfo["channelgroups"]
        self.badchans = badchans

    # def loadbasics(self):
    # pass

    def makerecinfo(self):

        myroot = ET.parse(self._obj.recfiles.xmlfile).getroot()

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
            "subname": self._obj.session.subname,
            "sessionName": self._obj.session.sessionName,
            "lfpSrate": 1250,
        }

        np.save(self._obj.files.basics, basics)
        print(f"_basics.npy created for {self._obj.session.sessionName}")
