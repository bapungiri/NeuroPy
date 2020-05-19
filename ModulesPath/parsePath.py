import os
from pathlib import Path
import pandas as pd
import numpy as np
import tkinter as tk


class path2files:
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
        self.spindles = Path(str(f_prefix) + "_spindles.npy")

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
