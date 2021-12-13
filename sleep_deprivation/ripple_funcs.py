import numpy as np
from neuropy.core import Epoch, Neurons


def ripple_modulation(neurons: Neurons, starts, peaks, stops, n_bins=4):
    """Neurons firing rate modulation within ripples
    Each ripple is divided into following sub-epochs of size n_bins:
        before-start, start-peak, peak-stop, stop-post

    Parameters
    ----------
    neurons : Neurons
        [description]
    starts : array
        ripple start times in seconds
    peaks : array
        ripple peak times in seconds
    stops : array
        ripple
    nbins : int, optional
        number of bins, by default 4

    Returns
    -------
    array: n_neurons x (4*n_bins)

    References
    ----------
    1. Diba et al. 2014
    2. Cscisvari et al. 1999
    """

    start_peak_dur, peak_stop_dur = peaks - starts, stops - peaks
    pre_start = Epoch.from_array(starts - start_peak_dur, starts)
    start_peak = Epoch.from_array(starts, peaks)
    peak_stop = Epoch.from_array(peaks, stops)
    stop_post = Epoch.from_array(stops, stops + peak_stop_dur)

    get_modulation = lambda e: neurons.get_modulation_in_epochs(e, n_bins)

    modulation = []
    for s in range(3):
        epoch_slices = [_[s::3] for _ in (pre_start, start_peak, peak_stop, stop_post)]
        modulation.append(np.hstack([get_modulation(_) for _ in epoch_slices]))
    modulation = np.dstack(modulation).sum(axis=2)

    return modulation
