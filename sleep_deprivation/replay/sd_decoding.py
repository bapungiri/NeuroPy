import numpy as np
import pandas as pd
import subjects
from neuropy.analyses import Pf1D
from neuropy.analyses import Decode1d

sessions = subjects.nsd.ratUday2 + subjects.sd.ratUday4

for sub, sess in enumerate(sessions):
    print(sess.animal.name)
    maze = sess.paradigm["maze"].flatten()
    post = sess.paradigm["post"].flatten()
    neurons = sess.neurons.get_neuron_type(neuron_type="pyr")
    pos = sess.maze
    # run = sess.run
    pf = Pf1D(
        neurons=neurons,
        position=pos,
        speed_thresh=4,
        sigma=4,
        grid_bin=2,
        # epochs=run,
        frate_thresh=1,
    )
    pf_neurons = neurons.get_by_id(pf.neuron_ids)
    epochs = sess.pbe.time_slice(post[0], post[0] + 2 * 3600)
    decode = Decode1d(
        neurons=pf_neurons,
        ratemap=pf,
        epochs=epochs,
        bin_size=0.02,
        decode_margin=15,
        nlines=5000,
    )
    decode.n_jobs = 5
    decode.calculate_shuffle_score(method="neuron_id", n_iter=200)

    df = pd.DataFrame(
        {
            "is_replay": decode.p_value < 0.05,
            "is_forward": decode.velocity > 0,
            "is_reverse": decode.velocity < 0,
            "p_value": decode.p_value,
            "score": decode.score,
            "velocity": decode.velocity,
            "intercept": decode.intercept,
        }
    )
    new_epochs = epochs.add_dataframe(df)
    new_epochs.metadata = {
        "posterior": decode.posterior,
        "shuffle_score": decode.shuffle_score,
    }
    new_epochs.save(sess.filePrefix.with_suffix(".replay.pbe"))
