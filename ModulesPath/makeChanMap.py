import os
import numpy as np
import xml.etree.ElementTree as ET


class ExtractChanXml:
    def __init__(self, basePath):
        self.sessionName = basePath.split("/")[-3] + basePath.split("/")[-2]
        print(self.sessionName)
        self.basePath = basePath
        for file in os.listdir(basePath):
            if file.endswith(".xml"):

                self.subname = file[:-4]
                self.filename = os.path.join(basePath, file)
                self.filePrefix = os.path.join(basePath, file[:-4])

        myroot = ET.parse(self.filename).getroot()

        self.chan_session = []
        for x in myroot.findall("anatomicalDescription"):
            for y in x.findall("channelGroups"):
                for z in y.findall("group"):
                    for chan in z.findall("channel"):
                        self.chan_session.append(int(chan.text))

        for sf in myroot.findall("acquisitionSystem"):
            self.sampfreq = int(sf.find("samplingRate").text)

        self.nChans = len(self.chan_session)

        basics = {
            "sRate": self.sampfreq,
            "channels": self.chan_session,
            "nChans": self.nChans,
        }

        np.save(self.filePrefix + "_basics.npy", basics)

