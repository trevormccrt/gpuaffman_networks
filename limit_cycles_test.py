import numpy as np

import binary_core, limit_cycles, general_network


def test_simple_cycle_lengths():
    xor_function = np.array([False, True, True, False])
    and_function = np.array([False, False, False, True])
    connectivity = np.array([[0, 1], [0, 2], [1, 2]])
    functions = np.array(np.stack([xor_function, and_function, xor_function], axis=0))
    init_states = np.array(binary_core.truth_table_columns(3))
    true_cycle_lengths = np.array([1, 1, 3, 1, 1, 3, 1, 3])
    true_cycles = np.array([[0, 1], [0, 1], [0, 3], [1, 2], [0, 1], [0, 3], [1, 2], [0, 3]])
    cycle_lengths, cycles, num_left, steps_used, _ = limit_cycles.measure_limit_cycle_lengths(init_states,functions, connectivity)
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
    cycle_lengths, _, num_left, steps_used, _ = limit_cycles.measure_limit_cycle_lengths(init_states, functions, connections)
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


def test_found_cycle_validity():
    batch_size = 100
    N = 30
    k = 2
    functions = binary_core.random_binary_data((batch_size, N, 1<<k), 0.5)
    connections = np.random.randint(0, N, (batch_size, N, k))
    init_states = binary_core.random_binary_data((batch_size, N), 0.5)
    cycle_lengths, cycle_start_end, num_left, steps_used, cycle_data = limit_cycles.measure_limit_cycle_lengths(init_states, functions, connections)
    assert np.all(cycle_lengths == [x.shape[0] for x in cycle_data])
    for function, connection, cycle, cycle_length in zip(functions, connections, cycle_data, cycle_lengths):
        current_state = cycle[0, :]
        for i in range(cycle_length):
            new_state = general_network.state_update(current_state, function, connection)
            if i < np.shape(cycle)[0] - 1:
                assert np.all(new_state == cycle[i+1])
            else:
                assert np.all(new_state == cycle[0])
            current_state = new_state

