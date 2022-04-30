import numpy as np


def get_jump_distance(arrs, estimator="max"):

    n_pos = arrs[0].shape[0]
    # norm_pos = np.linspace(0, 1, n_pos)
    dpos = 1 / n_pos

    match estimator:
        case 'mean': 
            return np.abs([np.mean(np.diff(np.argmax(_, axis=0)) * dpos) for _ in arrs])
        case 'median': 
            return np.abs([np.median(np.diff(np.argmax(_, axis=0)) * dpos) for _ in arrs])
        case 'max': 
            return np.abs([np.max(np.diff(np.argmax(_, axis=0)) * dpos) for _ in arrs])


