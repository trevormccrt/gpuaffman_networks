"""Dynamics for a general Boolean Network with N nodes and k connections per node."""
import numpy as np

from gpuaffman_networks import binary_core


def get_update_inputs(states, connections):
    """Get the inputs to a state update.

    Does the "wiring" step of the Boolean network state update. Given network states in `states` and the inputs
    to each node specified by `connections` organize the network states into a tensor in which the last dimension
    holds the values needed to update the state of each node.

    Args:
        states: A binary tensor with shape [..., N], representing the states of batches of N node Boolean networks.
        connections: An integer tensor containing values on [0, N-1] with a final dimension length k,
            broadcastable to the shape of `states`. Represents the inputs to each node of a Boolean network.
    Returns:
        A binary tensor with shape [..., N, k], where the values along the last dimension are the state values used to
            update the corresponding node.
    """
    return np.take_along_axis(np.expand_dims(states, -1),
                              np.broadcast_to(connections, (*states.shape, connections.shape[-1])), -2)


def state_update(states, functions, connections):
    """Get the updated state of a Boolean network.

    Use `functions` and `connections` to advance `states` one timestep.

    Args:
        states: as in get_update_inputs.
        functions: as in apply_binary_function.
        connections: as in get_update_inputs.
    Returns:
        A binary tensor with the same shape as `states` representing the updated state.
    """
    return binary_core.apply_binary_function(get_update_inputs(states, connections), functions)
