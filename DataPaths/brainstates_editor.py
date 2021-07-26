import ephyviewer

def editor(self, chan, spikes=None):
    class StatesSource(ephyviewer.WritableEpochSource):
        def __init__(
            self,
            filename,
            possible_labels,
            color_labels=None,
            channel_name="",
            restrict_to_possible_labels=False,
        ):

            self.filename = filename

            ephyviewer.WritableEpochSource.__init__(
                self,
                epoch=None,
                possible_labels=possible_labels,
                color_labels=color_labels,
                channel_name=channel_name,
                restrict_to_possible_labels=restrict_to_possible_labels,
            )

        def load(self):
            """
            Returns a dictionary containing the data for an epoch.
            Data is loaded from the CSV file if it exists; otherwise the superclass
            implementation in WritableEpochSource.load() is called to create an
            empty dictionary with the correct keys and types.
            The method returns a dictionary containing the loaded data in this form:
            { 'time': np.array, 'duration': np.array, 'label': np.array, 'name': string }
            """

            if self.filename.is_file():
                # if file already exists, load previous epoch
                data = pd.read_pickle(self.filename)
                state_number_dict = {1: "nrem", 2: "rem", 3: "quiet", 4: "active"}
                data["name"] = data["state"].map(state_number_dict)

                epoch_labels = np.array([f" State{_}" for _ in data["state"]])
                epoch = {
                    "time": data["start"].values,
                    "duration": data["end"].values - data["start"].values,
                    "label": epoch_labels,
                }
            else:
                # if file does NOT already exist, use superclass method for creating
                # an empty dictionary
                epoch = super().load()

            return epoch

        def save(self):
            df = pd.DataFrame()
            df["start"] = np.round(self.ep_times, 6)  # round to nearest microsecond
            df["end"] = np.round(self.ep_times, 6) + np.round(
                self.ep_durations
            )  # round to nearest microsecond
            df["duration"] = np.round(
                self.ep_durations, 6
            )  # round to nearest microsecond
            state_number_dict = {"nrem": 1, "rem": 2, "quiet": 3, "active": 4}
            df["name"] = self.ep_labels
            df["state"] = df["name"].map(state_number_dict)
            df.sort_values(["time", "duration", "name"], inplace=True)
            df.to_pickle(self.filename)

    states_source = StatesSource(self.files.states, self.labels)
    # you must first create a main Qt application (for event loop)
    # app = ephyviewer.mkQApp()

    sigs = np.asarray(self._obj.geteeg(chans=chan)).reshape(-1, 1)
    filtered_sig = signal_process.filter_sig.bandpass(
        sigs, lf=120, hf=150, ax=0, fs=1250
    )
    sample_rate = self._obj.lfpSrate
    t_start = 0.0

    # Create the main window that can contain several viewers
    win = ephyviewer.MainViewer(debug=True, show_auto_scale=True)

    # create a viewer for signal
    view1 = ephyviewer.TraceViewer.from_numpy(
        np.hstack((sigs, filtered_sig)), sample_rate, t_start, "Signals"
    )
    view1.params["scale_mode"] = "same_for_all"
    view1.auto_scale()
    win.add_view(view1)

    source_sig = ephyviewer.InMemoryAnalogSignalSource(sigs, sample_rate, t_start)
    # create a viewer for the encoder itself
    view2 = ephyviewer.EpochEncoder(
        source=states_source, name="Dev mood states along day"
    )
    win.add_view(view2)

    view3 = ephyviewer.TimeFreqViewer(source=source_sig, name="tfr")
    view3.params["show_axis"] = False
    view3.params["timefreq", "deltafreq"] = 1
    win.add_view(view3)

    # ----- spikes --------
    if spikes is not None:
        spk_id = np.arange(len(spikes))

        all_spikes = []
        for i, (t, id_) in enumerate(zip(spikes, spk_id)):
            all_spikes.append({"time": t, "name": f"Unit {i}"})

        spike_source = ephyviewer.InMemorySpikeSource(all_spikes=all_spikes)
        view4 = ephyviewer.SpikeTrainViewer(source=spike_source)
        win.add_view(view4)
        # show main window and run Qapp
    # win.show()
    # return win, app

    # app.exec_()

    return win
