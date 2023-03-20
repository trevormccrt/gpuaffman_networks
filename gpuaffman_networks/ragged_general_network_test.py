import cupy as cp
import numpy as np


import general_network, ragged_general_network


def test_ragged_k_state_update_extra_dims():
    batch_size = 100
    N = 15
    k_true = 3
    k_cont = 5
    states = np.random.binomial(1, 0.5, (batch_size, N)).astype(np.bool_)
    true_connectivity = np.random.randint(0, N, (batch_size, N, k_true))
    true_functions = np.random.binomial(1, 0.5, (batch_size, N, 1<<k_true)).astype(np.bool_)
    true_update = general_network.state_update(states, true_functions, true_connectivity)
    extra_connectivity = np.random.randint(0, 10, (batch_size, N, k_cont - k_true))
    total_connectivity = np.concatenate([true_connectivity, extra_connectivity], axis=-1)
    extra_functions = np.random.binomial(1, 0.5, (batch_size, N, 1<<k_cont - 1<<k_true)).astype(np.bool_)
    total_functions = np.concatenate([true_functions, extra_functions], axis=-1)
    used_connections = np.concatenate([np.ones((batch_size, N, k_true)), np.zeros((batch_size, N, k_cont - k_true))], axis=-1).astype(np.bool_)
    ragged_update = ragged_general_network.ragged_k_state_update(states, total_functions, total_connectivity, used_connections)
    np.testing.assert_equal(true_update, ragged_update)


def test_ragged_k_state_update_mixed_dims():
    batch_size = 100
    N = 20
    k_vals = [3, 5]
    states = np.random.binomial(1, 0.5, (batch_size, 2, N)).astype(np.bool_)
    functions = np.random.binomial(1, 0.5, (batch_size, 2, N, 1 << np.max(k_vals))).astype(np.bool_)
    connections = np.random.randint(0, N, (batch_size, 2, N, np.max(k_vals)))
    used_connections = np.ones((batch_size, 2, N, np.max(k_vals))).astype(np.bool_)
    used_connections[:, 0, :, np.min(k_vals):] = False
    ragged_update = ragged_general_network.ragged_k_state_update(states, functions, connections, used_connections)
    for i, k in enumerate(k_vals):
        this_state = states[:, i, :]
        this_connections = connections[:, i, :, :k]
        this_functions = functions[:, i, :, : 1 << k]
        normal_update = general_network.state_update(this_state, this_functions, this_connections)
        np.testing.assert_equal(normal_update, ragged_update[:, i, :])


def test_cuda():
    batch_size = 100
    N = 8
    k_max = 8
    states = np.random.binomial(1, 0.5, (batch_size, 2, N)).astype(np.bool_)
    functions = np.random.binomial(1, 0.5, (batch_size, 2, N, 1 << k_max)).astype(np.bool_)
    connections = np.random.randint(0, N, (batch_size, 2, N, k_max))
    used_connections = np.ones((batch_size, 2, N, k_max)).astype(np.bool_)
    result_np = ragged_general_network.ragged_k_state_update(states, functions, connections, used_connections)
    result_cp = ragged_general_network.ragged_k_state_update(cp.array(states), cp.array(functions), cp.array(connections), cp.array(used_connections))
    np.testing.assert_equal(result_np, cp.asnumpy(result_cp))
