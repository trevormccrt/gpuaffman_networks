"""Dynamics for a ragged Boolean Network with N nodes k or less connections per node."""
import numpy as np

from gpuaffman_networks import binary_core, general_network


def apply_ragged_k_binary_function(binary_data, functions, used_connections):
    """Apply an arbitrary ragged binary function to some data.

    Apply an arbitrary binary function to the last axis of `binary_data`. The function(s) to apply are specified in the
    form of truth tables by `functions`, the batch shape of which must be broadcastable to the shape of `binary_data`.
    `used_connections` will be used to mask values in `binary_data`, such that only values that correspond to the value
    1 in `used_connections` will contribute.

    Args:
        binary_data: a binary tensor with shape [..., d].
        functions: a binary tensor with final dimension lengths 2^d. The other dimensions have to be broadcastable to
            the shape of `binary_data`.
        used_connections:
    Returns:
        A tensor of shape [...], containing the result of applying `functions` to `binary_data`.
    """
    return np.squeeze(np.take_along_axis(np.broadcast_to(functions, (*binary_data.shape[:-1], functions.shape[-1])),
                                         np.expand_dims(binary_core.binary_to_uint8(
                                             np.bitwise_and(binary_data, used_connections)), -1), -1), -1)


def ragged_k_state_update(states, functions, connections, used_connections):
    """Get the updated state of a ragged Boolean network.

    Use `functions`, `connections`, and `used_connections` to advance `states` one timestep.

    Args:
        states: as in get_update_inputs.
        functions: as in apply_ragged_k_binary_function.
        connections: as in get_update_inputs.
        used_connections: as in apply_ragged_k_binary_function
    Returns:
        A binary tensor with the same shape as `states` representing the updated state.
    """
    return apply_ragged_k_binary_function(general_network.get_update_inputs(states, connections),
                                          functions, used_connections)
