"""
ephyviewer also provides an epoch encoder which can be used with shortcut keys
and/or the mouse to encode labels.

ephyviewer makes available a CsvEpochSource class, which inherits from
WritableEpochSource. If you would like to customize reading and writing epochs
to files, you can write your own subclass of WritableEpochSource that implements
the load() and save() methods.

Here is an example of an epoch encoder that uses CsvEpochSource.

"""

from ephyviewer import (
    mkQApp,
    MainViewer,
    TraceViewer,
    CsvEpochSource,
    EpochEncoder,
    WritableEpochSource,
    epochs,
)
import numpy as np
import subjects

sess = subjects.Sd().ratNday1[0]
states = sess.brainstates.states
states_new = {
    "time": states.start.values,
    "duration": states.duration.values,
    "label": states.name.values,
}

possible_labels = ["nrem", "rem", "quiet", "active"]
source_epoch = WritableEpochSource(epoch=states_new)

maze = sess.epochs.maze
lfp = np.array(sess.recinfo.geteeg(chans=67)).reshape(-1, 1)


# lets encode some dev mood along the day

filename = "example_dev_mood_encoder.csv"
# source_epoch = CsvEpochSource(filename, possible_labels)


# you must first create a main Qt application (for event loop)
app = mkQApp()

# create fake 16 signals with 100000 at 10kHz
sigs = np.random.rand(100000, 16)
sample_rate = 1000.0
t_start = 0.0

# Create the main window that can contain several viewers
win = MainViewer(debug=True, show_auto_scale=True)

# create a viewer for signal
view1 = TraceViewer.from_numpy(lfp, 1250.0, t_start, "Signals")
view1.params["scale_mode"] = "same_for_all"
view1.auto_scale()
win.add_view(view1)

# create a viewer for the encoder itself
view2 = EpochEncoder(source=source_epoch, name="Dev mood states along day")
win.add_view(view2)


# show main window and run Qapp
win.show()
