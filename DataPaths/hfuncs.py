import signal_process
from scipy import stats
import numpy as np
import pandas as pd


def wavelet_gamma_theta_phase(signal, theta_phase, binsize=9, frgamma=None, fs=1250):

    if frgamma is None:
        frgamma = np.arange(25, 150)

    # ----- wavelet power for gamma oscillations----------
    wavdec = signal_process.wavelet_decomp(signal, freqs=frgamma, sampfreq=fs)
    wav = wavdec.colgin2009()
    wav = stats.zscore(wav, axis=1)

    # ----segmenting gamma wavelet at theta phases ----------
    bin_angle = np.linspace(0, 360, int(360 / binsize) + 1)
    phase_centers = bin_angle[:-1] + np.diff(bin_angle).mean() / 2

    bin_ind = np.digitize(theta_phase, bin_angle)

    gamma_at_theta = pd.DataFrame()
    for i in np.unique(bin_ind):
        find_where = np.where(bin_ind == i)[0]
        gamma_at_theta[phase_centers[i - 1]] = np.mean(wav[:, find_where], axis=1)
    gamma_at_theta.insert(0, column="freq", value=frgamma)
    gamma_at_theta.set_index("freq", inplace=True)

    return gamma_at_theta
