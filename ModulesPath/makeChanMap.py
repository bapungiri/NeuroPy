import os
import numpy as np
import xml.etree.ElementTree as ET
from parsePath import path2files


class recinfo(path2files):
    nShanks = 8

    def __init__(self, basePath):
        super().__init__(basePath)
        # self.__basics = path2files(basePath)

        myinfo = np.load(self._files.basics, allow_pickle=True).item()
        # print(recinfo.keys())
        self.sampfreq = myinfo["sRate"]
        self.channels = myinfo["channels"]
        self.nChans = myinfo["nChans"]
        self.lfpSrate = 1250
        # # self.channelgroups = recinfo["channelgroups"]

    # def loadbasics(self):
    # pass

    def makebasics(self):

        myroot = ET.parse(self.__basics.xmlfile).getroot()

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
            "subname": self.__basics.subname,
            "sessionName": self.__basics.sessionName,
            "lfpSrate": 1250,
        }

        np.save(self.__basics.files.basics, basics)
        print(f"_basics.npy created for {self.__basics.sessionName}")

