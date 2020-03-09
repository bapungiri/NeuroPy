import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

a = pd.Timestamp("2012-05-01 00:05:45")
b = pd.to_datetime(pd.Series(["2012-05-01 00:06:45", "2012-05-01 00:07:45"]))
