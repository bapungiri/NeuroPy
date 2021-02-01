import matplotlib.pyplot as plt
import numpy as np

# fig, ax = plt.subplots(1, 2)
fig = plt.figure()
axs = plt.subplot2grid((5, 5), (0, 0), sharex=True)

axs.plot([1, 2, 3])
