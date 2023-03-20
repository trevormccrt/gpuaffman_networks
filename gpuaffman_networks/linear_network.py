"""Dynamics for a Boolean network with linear topology and nearest-neighbour interactions."""
import numpy as np

import binary_core


def state_update(state, functions):
    """Get the updated state of a Boolean network.

    Use `functions` to advance `states` one timestep.

    Args:
        states: A binary tensor with shape [..., N], representing the states of batches of N node Boolean networks.
        functions: as in apply_binary_function.
    Returns:
        A binary tensor with the same shape as `states` representing the updated state.
    """
    state_rolled = np.stack([np.roll(state, 1, -1), state, np.roll(state, -1, -1)], axis=-1)
    return binary_core.apply_binary_function(state_rolled, functions)
