import numpy as np
import pandas as pd
import subjects
from neuropy.analyses import Pf1D
from neuropy.analyses import Decode1d

sessions = subjects.nsd.ratUday2

for sub, sess in enumerate(sessions):
    maze = sess.paradigm["maze"]
    post = sess.paradigm["post"]
    neurons = sess.neurons.get_neuron_type(neuron_type="pyr")
    pos = sess.lin_maze
    run = sess.run
    pf = Pf1D(
        neurons=neurons,
        position=pos,
        speed_thresh=4,
        sigma=4,
        grid_bin=2,
        # epochs=run,
        frate_thresh=1,
    )
    pf_neurons = neurons.get_by_id(pf.ratemap.neuron_ids)
    epochs = sess.pbe.time_slice(post[0], post[1])
    decode = Decode1d(
        neurons=pf_neurons, ratemap=pf.ratemap, epochs=epochs, bin_size=0.02
    )
    decode.n_jobs = 6
    decode.calculate_shuffle_score(method="neuron_id", n_iter=1000)

    df = pd.DataFrame(
        {
            "is_replay": decode.p_value < 0.01,
            "is_forward": (decode.slope > 0) & (decode.p_value < 0.01),
            "is_reverse": (decode.slope < 0) & (decode.p_value < 0.01),
            "replay_p_value": decode.p_value,
            "replay_score": decode.score,
            "replay_slope": decode.slope,
        }
    )
    new_epochs = epochs.add_column("replay_p_value", decode.p_value)
    new_epochs = epochs.add_dataframe(df)
    new_epochs.metadata = {"posterior": decode.posterior}
    new_epochs.filename = sess.filePrefix.with_suffix(".replay.pbe")
    new_epochs.save()
