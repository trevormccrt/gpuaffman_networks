import cupy as cp
import numpy as np


from gpuaffman_networks import general_network, linear_network


def test_prepare_input():
    batch_size = 100
    N = 30
    k = 5
    population = np.random.binomial(1, 0.5, (batch_size, N)).astype(np.bool_)
    connections = np.random.randint(0, N, (batch_size, N, k))
    wired = general_network.get_update_inputs(population, connections)
    for i in range(batch_size):
        for j in range(N):
            for h, wire_to in enumerate(connections[i, j]):
                assert wired[i, j, h] == population[i, wire_to]


def test_vs_linear_network():
    batch_size = 100
    N = 30
    functions = np.random.binomial(1, 0.5, (batch_size, N, 8)).astype(np.bool_)
    indexes = np.arange(start=0, stop=N, step=1)
    connections = np.array(np.broadcast_to(np.expand_dims(np.stack([np.roll(indexes, 1), indexes, np.roll(indexes, -1)], axis=-1), 0), (batch_size, N, 3)))
    states = np.random.binomial(1, 0.5, (batch_size, N)).astype(np.bool_)
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
