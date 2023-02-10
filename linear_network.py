import numpy as np

import binary_core


def state_update(state, function):
    state_rolled = np.stack([np.roll(state, 1, -1), state, np.roll(state, -1, -1)], axis=-1)
    return binary_core.apply_binary_function(state_rolled, function)