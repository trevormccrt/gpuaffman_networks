import cupy as cp
import numpy as np


import binary_core, general_network, ragged_general_network


def test_ragged_k_state_update_extra_dims():
    batch_size = 100
    N = 15
    k_true = 3
    k_cont = 5
    states = cp.array(binary_core.random_binary_data((batch_size, N), 0.5))
    true_connectivity = cp.array(np.random.randint(0, N, (batch_size, N, k_true)))
    true_functions = cp.array(binary_core.random_binary_data((batch_size, N, 1<<k_true), 0.5))
    true_update = general_network.state_update(states, true_functions, true_connectivity)
    extra_connectivity = cp.array(np.random.randint(0, 10, (batch_size, N, k_cont - k_true)))
    total_connectivity = cp.concatenate([true_connectivity, extra_connectivity], axis=-1)
    extra_functions = cp.array(binary_core.random_binary_data((batch_size, N, 1<<k_cont - 1<<k_true), 0.5))
    total_functions = cp.concatenate([true_functions, extra_functions], axis=-1)
    used_connections = cp.concatenate([cp.ones((batch_size, N, k_true)), cp.zeros((batch_size, N, k_cont - k_true))], axis=-1).astype(cp.bool_)
    ragged_update = ragged_general_network.ragged_k_state_update(states, total_functions, total_connectivity, used_connections)
    cp.testing.assert_array_equal(true_update, ragged_update)


def test_ragged_k_state_update_mixed_dims():
    batch_size = 100
    N = 20
    k_vals = [3, 5]
    states = cp.array(binary_core.random_binary_data((batch_size, 2, N), 0.5))
    functions = cp.array(binary_core.random_binary_data((batch_size, 2, N, 1 << np.max(k_vals)), 0.5))
    connections = cp.array(np.random.randint(0, N, (batch_size, 2, N, np.max(k_vals))))
    used_connections = cp.array(cp.ones((batch_size, 2, N, np.max(k_vals)))).astype(cp.bool_)
    used_connections[:, 0, :, np.min(k_vals):] = False
    ragged_update = ragged_general_network.ragged_k_state_update(states, functions, connections, used_connections)
    for i, k in enumerate(k_vals):
        this_state = states[:, i, :]
        this_connections = connections[:, i, :, :k]
        this_functions = functions[:, i, :, : 1 << k]
        normal_update = general_network.state_update(this_state, this_functions, this_connections)
        cp.testing.assert_array_equal(normal_update, ragged_update[:, i, :])
