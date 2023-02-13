import cupy as cp
import numpy as np


import binary_core, general_network, linear_network


def test_prepare_input():
    batch_size = 100
    N = 30
    k = 5
    population = binary_core.random_binary_data((batch_size, N), 0.5)
    connections = np.random.randint(0, N, (batch_size, N, k))
    wired = general_network.get_update_inputs(population, connections)
    for i in range(batch_size):
        for j in range(N):
            for h, wire_to in enumerate(connections[i, j]):
                assert wired[i, j, h] == population[i, wire_to]


def test_vs_linear_network():
    batch_size = 100
    N = 30
    functions = binary_core.random_binary_data((batch_size, N, 8), 0.5)
    indexes = np.arange(start=0, stop=N, step=1)
    connections = np.array(np.broadcast_to(np.expand_dims(np.stack([np.roll(indexes, 1), indexes, np.roll(indexes, -1)], axis=-1), 0), (batch_size, N, 3)))
    states = binary_core.random_binary_data((batch_size, N), 0.5)
    general_update = general_network.state_update(states, functions, connections)
    linear_update = linear_network.state_update(states, functions)
    np.testing.assert_equal(general_update, linear_update)


def test_basic_nonlin_network():
    xor_function = np.array([False, True, True, False])
    and_function = np.array([False, False, False, True])
    connectivity = np.array([[0, 1], [0, 2], [1, 2]])
    functions = np.stack([xor_function, and_function, xor_function], axis=0)
    state = np.array([True, False, True])
    all_states = np.tile(np.expand_dims(state, 0), (4, 1))
    for i in range(3):
        all_states[i+1, :] = general_network.state_update(all_states[i], functions, connectivity)
    np.testing.assert_equal(all_states, np.array([[True, False, True], [True, True, True], [False, True, False], [True, False, True]]))


def test_cuda():
    batch_size = 100
    N = 8
    k = 3
    state = np.random.binomial(1, 0.5, (batch_size, N)).astype(np.bool_)
    functions = np.random.binomial(1, 0.5, (batch_size, N, 1<<k)).astype(np.bool_)
    connectivity = np.random.randint(0, N, (batch_size, N, k)).astype(np.uint8)
    state_np = general_network.state_update(state, functions, connectivity)
    state_cp = general_network.state_update(cp.array(state), cp.array(functions), cp.array(connectivity))
    np.testing.assert_equal(state_np, cp.asnumpy(state_cp))


def test_simple_cycle_lengths():
    xor_function = np.array([False, True, True, False])
    and_function = np.array([False, False, False, True])
    connectivity = np.array([[0, 1], [0, 2], [1, 2]])
    functions = np.array(np.stack([xor_function, and_function, xor_function], axis=0))
    init_states = np.array(binary_core.truth_table_columns(3))
    true_cycle_lengths = np.array([1, 1, 3, 1, 1, 3, 1, 3])
    true_cycles = np.array([[0, 1], [0, 1], [0, 3], [1, 2], [0, 1], [0, 3], [1, 2], [0, 3]])
    cycle_lengths, cycles, num_left, steps_used = general_network.measure_limit_cycle_lengths(init_states,functions, connectivity)
    assert steps_used < 8
    assert not num_left
    np.testing.assert_equal(true_cycle_lengths, cycle_lengths)
    np.testing.assert_equal(true_cycles, cycles)


def test_complex_cycle_lengths():
    batch_size = 100
    N = 30
    k = 2
    functions = binary_core.random_binary_data((batch_size, N, 1<<k), 0.5)
    connections = np.random.randint(0, N, (batch_size, N, k))
    init_states = binary_core.random_binary_data((batch_size, N), 0.5)
    cycle_lengths, _, num_left, steps_used = general_network.measure_limit_cycle_lengths(init_states, functions, connections)
    assert not num_left
    n_sim_timesteps = int(1.5 * steps_used)
    all_states = np.moveaxis(np.tile(np.expand_dims(np.copy(init_states), -1), n_sim_timesteps+1), -1, 0)
    for i in range(n_sim_timesteps):
        all_states[i+1, ...] = general_network.state_update(all_states[i, ...], functions, connections)
    for i in range(batch_size):
        for t in range(n_sim_timesteps):
            current_state = all_states[t, i, :]
            compare_states = all_states[:t, i, :]
            matches = np.where(np.all(np.equal(current_state, compare_states), axis=-1))
            matches = np.array(matches)
            if np.any(matches):
                cycle_length = t - matches[0][0]
                assert cycle_length == cycle_lengths[i]
                break
        else:
            raise Exception

