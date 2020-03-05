import os
import numpy as np
import xml.etree.ElementTree as ET
from parsePath import name2path


class ExtractChanXml(name2path):
    nShanks = 8

    def __init__(self, basePath):
        super().__init__(basePath)

    def makebasics(self):

        myroot = ET.parse(self.xmlfile).getroot()

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
            "subname": self.subname,
            "sessionName": self.sessionName,
        }

        np.save(str(self.filePrefix) + "_basics.npy", basics)
        print(f"_basics.npy created for {self.sessionName}")

