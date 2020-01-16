import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# folderPath = '../'

filename = 'template.prm'
session = 'RatJDay1'  


with open(filename) as f:
    with open("out.txt", "w") as f1:
        for line in f:
            print(line)
            if "experiment_name" in line:
                f1.write("experiment_name = " + session + '\n')
            if "prb_file" in line:
                f1.write("n_channels = " + session + '.prb\n')
            # if "raw_data_files" in line:
            #     f1.write("raw_data_files = " + session + '.dat\n')
            # if "n_channels" in line:
            #     f1.write("n_channels = " + str(32) + ',\n')
                
            else:
                f1.write(line)
                print(line)