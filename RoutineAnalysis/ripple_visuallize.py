import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib
from bokeh.plotting import output_file, show, save
from bokeh.layouts import row, column
from bokeh.models import Label
from bokeh.layouts import gridplot
from matplotlib.gridspec import GridSpec


# plot.output_backend = "svg"
output_file("vline_stack.html")

# mpl.style.use("figPublish")


from test2 import plotcheck


m = plotcheck()

fig = plt.figure()
plt.subplot(1, 2, 2)
m.draw

fig.show()



from callfunc import processData

# TODO thoughts on using data class for loading data into function

basePath = [
    "/data/Clustering/SleepDeprivation/RatJ/Day1/",
    "/data/Clustering/SleepDeprivation/RatJ/Day2/",
    # "/data/Clustering/SleepDeprivation/RatJ/Day3/",
    # "/data/Clustering/SleepDeprivation/RatJ/Day4/",
    "/data/Clustering/SleepDeprivation/RatK/Day1/",
    "/data/Clustering/SleepDeprivation/RatK/Day2/",
    "/data/Clustering/SleepDeprivation/RatN/Day1/",
    "/data/Clustering/SleepDeprivation/RatN/Day2/",
]


sessions = [processData(_) for _ in basePath]

#
plt.close("all")
fig = plt.figure(figsize=(6, 10))
gs = GridSpec(6, 2, figure=fig)
for i, sess in enumerate(sessions):
    # sess.recinfo.makerecinfo()
    # sess.trange = np.array([])
    # sess.ripple.findswr()

    #%% --- Ripple power block -------------

    # sess.trange = sess.epochs.pre
    # sess.eventpsth.hswa_ripple.nQuantiles = 5
    # p1, _ = sess.eventpsth.hswa_ripple.plot()
    # mytext = Label(x=10, y=170, x_units="screen", y_units="screen", text="PRE")
    # p1.add_layout(mytext)

    # sess.trange = np.asarray([sess.epochs.post[0] + 5 * 3600, sess.epochs.post[1]])
    # # sess.eventpsth.hswa_ripple.nQuantiles = 5

    # p2, _ = sess.eventpsth.hswa_ripple.plot()
    # mytext = Label(x=10, y=170, x_units="screen", y_units="screen", text="POST")
    # p2.add_layout(mytext)

    # p1.toolbar.logo = None
    # p1.toolbar_location = None
    # p2.toolbar.logo = None
    # p2.toolbar_location = None
    # fig.append([p1, p2])

    #%%===== Ripple peakpower over time ================

    sess.trange = sess.epochs.pre
    ripp_power = sess.ripple.peakpower
    ripp_time = sess.ripple.time[:, 0]

    ax1 = fig.add_subplot(gs[i, 0])
    ax1.plot(ripp_time / 3600, ripp_power, ".", markersize=0.8, color="#cd7070")
    ax1.set_title(sess.sessinfo.session.sessionName, x=0.2, y=0.90)
    ax1.set_ylabel("peakpower")

    sess.trange = np.asarray([sess.epochs.post[0] + 5 * 3600, sess.epochs.post[1]])
    ripp_power = sess.ripple.peakpower
    ripp_time = sess.ripple.time[:, 0]
    ax2 = fig.add_subplot(gs[i, 1])
    ax2.plot(ripp_time / 3600, ripp_power, ".", markersize=0.8, color="#83ce89")

    # fig.append(temp)

ax1.set_xlabel("Time (h)")
ax2.set_xlabel("Time (h)")

fig.show()

# for f in fig:
#     f.ax1()
# p = gridplot(fig, toolbar_location=None, plot_width=400, plot_height=200)

# save(p)
# fig1 = plt.figure()
# fig1.add
# #  axes.append(fig[0])

# plt.show()
