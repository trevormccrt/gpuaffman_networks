import cupy as cp

from boolean_networks import cuda_binary_core, general_network


def apply_ragged_k_binary_function(binary_data, functions, used_connections):
    return cp.squeeze(cp.take_along_axis(cp.broadcast_to(functions, (*binary_data.shape[:-1], functions.shape[-1])),
                                         cp.expand_dims(cuda_binary_core.binary_to_uint8(
                                             cp.bitwise_and(binary_data, used_connections)), -1), -1), -1)


def ragged_k_state_update(states, functions, connections, used_connections):
    return apply_ragged_k_binary_function(general_network.get_update_inputs(states, connections),
                                          functions, used_connections)
