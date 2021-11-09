from ephyviewer import mkQApp, MainViewer, TraceViewer
import numpy as np
from neuropy.core import Signal, ProbeGroup

# you must first create a main Qt application (for event loop)
def view_signal(signal: Signal, probegroup: ProbeGroup):
    app = mkQApp()

    # create fake 16 signals with 100000 at 10kHz

    sigs = signal.traces.T
    sample_rate = signal.sampling_rate
    t_start = signal.t_start
    sig_chan_ids = signal.channel_id

    shank_ids = probegroup.shank_id
    probe_chan_ids = probegroup.channel_id

    # sort channels based on shank ids
    sig_shank_ids = shank_ids[np.isin(probe_chan_ids, sig_chan_ids)]
    sort_ind = np.argsort(sig_shank_ids)
    sigs = sigs[:, sort_ind]

    # Create the main window that can contain several viewers
    win = MainViewer(debug=True, show_auto_scale=True)

    # create a viewer for signal with TraceViewer
    # TraceViewer normally accept a AnalogSignalSource but
    # TraceViewer.from_numpy is facitilty function to bypass that
    view1 = TraceViewer.from_numpy(sigs, sample_rate, t_start, "Signals")

    # Parameters can be set in script
    view1.params["scale_mode"] = "same_for_all"
    view1.params["display_labels"] = True

    # And also parameters for each channel
    # view1.by_channel_params["ch0", "visible"] = False
    # view1.by_channel_params["ch15", "color"] = "#FF00AA"

    # This is needed when scale_mode='same_for_all'
    # to recompute the gain
    # this avoid to push auto_scale button
    view1.auto_scale()

    # put this veiwer in the main window
    win.add_view(view1)

    # show main window and run Qapp
    win.show()

    app.exec_()


def view_multiple_signals(signal_list: list[Signal], names=None):
    app = mkQApp()

    n_sigs = len(signal_list)
    if names is None:
        names = [f"Signal{_}" for _ in range(n_sigs)]

    assert len(names) == n_sigs, "names and signal_list should have same length"

    win = MainViewer(debug=False, show_auto_scale=True)

    for i, sig in enumerate(signal_list):
        view1 = TraceViewer.from_numpy(
            sig.traces.T, sig.sampling_rate, sig.t_start, name=names[i]
        )

        view1.params["scale_mode"] = "same_for_all"
        view1.params["display_labels"] = True

        view1.auto_scale()

        # put this veiwer in the main window
        win.add_view(view1)

    # show main window and run Qapp
    win.show()
    app.exec_()
