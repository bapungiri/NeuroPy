import numpy as np

file = "/data/Clustering/SleepDeprivation/RatK/Day4/RatK_2019-08-16_04-42-36/experiment1/recording1/events/Message_Center-904.0/TEXT_group_1/text.npy"

data = np.load(file)

time_file = "/data/Clustering/SleepDeprivation/RatK/Day4/RatK_2019-08-16_04-42-36/experiment1/recording1/events/Message_Center-904.0/TEXT_group_1/timestamps.npy"

time_data = np.load(time_file)
