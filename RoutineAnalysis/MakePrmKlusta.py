import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# folderPath = '../'


class makePrmPrb:

    nChans = 64

    def __init__(self, basePath):
        self.sessionName = basePath.split("/")[-3] + basePath.split("/")[-2]
        print(self.sessionName)
        self.basePath = basePath
        for file in os.listdir(basePath):
            if file.endswith(".dat"):

                self.subname = file[:-4]
                self.filename = os.path.join(basePath, file)
                self.filePrefix = os.path.join(basePath, file[:-4])
        self.prmTemplate = (
            "/home/bapung/Documents/MATLAB/pythonprogs/RoutineAnalysis/template.prm"
        )
        self.prbTemplate = (
            "/home/bapung/Documents/MATLAB/pythonprogs/RoutineAnalysis/template.prb"
        )

        recInfo = np.load(self.filePrefix + "_basics.npy", allow_pickle=True)
        self.sampFreq = recInfo.item().get("sRate")
        self.chan_session = recInfo.item().get("channels")
        self.nChansDat = recInfo.item().get("nChans")

    def makePrm(self):
        for shank in range(1, 9):
            with open(self.prmTemplate) as f:
                if not os.path.exists(self.basePath + "Shank" + str(shank)):
                    os.mkdir(self.basePath + "Shank" + str(shank))
                outfile_prefix = (
                    self.basePath
                    + "Shank"
                    + str(shank)
                    + "/"
                    + self.sessionName
                    + "sh"
                    + str(shank)
                )

                with open(outfile_prefix + ".prm", "w") as f1:
                    for line in f:

                        if "experiment_name" in line:

                            f1.write(
                                "experiment_name = '"
                                + self.sessionName
                                + "sh"
                                + str(shank)
                                + "'\n"
                            )
                        elif "prb_file" in line:
                            f1.write("prb_file = '" + outfile_prefix + ".prb'\n")
                        elif "raw_data_files" in line:
                            f1.write(
                                "   raw_data_files = ['" + self.filePrefix + ".dat'],\n"
                            )
                        elif "sample_rate" in line:
                            f1.write("   sample_rate = " + str(self.sampFreq) + ",\n")
                        elif "n_channels" in line:
                            f1.write("  n_channels = " + str(self.nChansDat) + ",\n")

                        else:
                            f1.write(line)

    def makePrb(self):
        for shank in range(1, 9):

            chan_start = (shank - 1) * 8
            chan_end = chan_start + 8
            chan_list = self.chan_session[chan_start:chan_end]
            # chan_list = np.arange(chan_start, chan_end).tolist()
            with open(self.prbTemplate) as f:
                if not os.path.exists(self.basePath + "Shank" + str(shank)):
                    os.mkdir(self.basePath + "Shank" + str(shank))
                outfile_prefix = (
                    self.basePath
                    + "Shank"
                    + str(shank)
                    + "/"
                    + self.sessionName
                    + "sh"
                    + str(shank)
                )

                with open(outfile_prefix + ".prb", "w") as f1:
                    for line in f:

                        if "Shank index" in line:
                            f1.write("# Shank index. \n")
                            f1.write(str(shank - 1) + ":\n")
                            next(f)

                        elif "channels" in line:
                            f1.write("'channels' : " + str(chan_list) + ",")

                        elif "graph" in line:
                            f1.write("'graph' : [\n")
                            # f1.write("(" +str(chan_list[0])',' +")")
                            f1.write(str(tuple([chan_list[x] for x in [0, 1]])) + ",\n")
                            f1.write(str(tuple([chan_list[x] for x in [0, 2]])) + ",\n")
                            f1.write(str(tuple([chan_list[x] for x in [1, 2]])) + ",\n")
                            f1.write(str(tuple([chan_list[x] for x in [1, 3]])) + ",\n")
                            f1.write(str(tuple([chan_list[x] for x in [2, 3]])) + ",\n")
                            f1.write(str(tuple([chan_list[x] for x in [2, 4]])) + ",\n")
                            f1.write(str(tuple([chan_list[x] for x in [3, 4]])) + ",\n")
                            f1.write(str(tuple([chan_list[x] for x in [3, 5]])) + ",\n")
                            f1.write(str(tuple([chan_list[x] for x in [4, 5]])) + ",\n")
                            f1.write(str(tuple([chan_list[x] for x in [4, 6]])) + ",\n")
                            f1.write(str(tuple([chan_list[x] for x in [5, 6]])) + ",\n")
                            f1.write(str(tuple([chan_list[x] for x in [5, 7]])) + ",\n")
                            f1.write(str(tuple([chan_list[x] for x in [6, 7]])) + ",\n")

                            for i in range(13):
                                next(f)

                        elif "geometry" in line:
                            f1.write("'geometry' : {\n")
                            # f1.write("(" +str(chan_list[0])',' +")")
                            chan_height = np.arange(300, 140, -20)
                            for i in range(8):
                                f1.write(
                                    str(chan_list[i])
                                    + ":"
                                    + str((0, chan_height[i]))
                                    + ",\n"
                                )

                            for i in range(8):
                                next(f)

                        else:
                            f1.write(line)


filename = "template.prm"
basePath = "/data/Clustering/SleepDeprivation/RatJ/Day3/"


RatJDay3 = makePrmPrb(basePath)
RatJDay3.makePrm()
RatJDay3.makePrb()

