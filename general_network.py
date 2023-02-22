import numpy as np


import binary_core


def get_update_inputs(states, connections):
    connections = np.broadcast_to(connections, (*states.shape, connections.shape[-1]))
    return np.take_along_axis(np.expand_dims(states, -1), connections, -2)


def state_update(states, functions, connections):
    return binary_core.apply_binary_function(get_update_inputs(states, connections), functions)
