import sys
import os
sys.path.append(os.path.join(os.getenv("HOME"), "gpuaffman_networks/"))

import cupy as cp
import datetime
import numpy as np

import pickle

import cupy.cuda.device

import ragged_general_network, genetics_util


os.environ["CUPY_ACCELERATORS"] = "cutensor"

cupy.cuda.device.Device(0).use()
mempool = cupy.get_default_memory_pool()

def make_memory_input_state(N):
    input_state = cp.zeros((2, N), dtype=cp.uint8)
    input_state[1, 0] = True
    return input_state


def evaluate_memory_task(data):
    error_rate_false = cp.mean(cp.equal(data[:, 0, :, :, 0], False), axis=0)
    error_rate_true = cp.mean(cp.equal(data[:, 1, :, :, 0], True), axis=0)
    return cp.mean(cp.stack([error_rate_false, error_rate_true], axis=-1), axis=-1)


def run_dynamics_forward(input_state, functions, connections, used_connections, n_timesteps, p_noise):
    current_state = input_state
    for _ in range(n_timesteps):
        current_state = cp.bitwise_xor(ragged_general_network.ragged_k_state_update(current_state, functions, connections, used_connections), cp.random.binomial(1, p_noise, current_state.shape, dtype=cp.bool_))
    return ragged_general_network.ragged_k_state_update(current_state, functions, connections, used_connections)


def split_breeding(first_parents, second_parents, n_children):
    mix_children = int(n_children/2)
    return cp.concatenate([genetics_util.pair_breed_swap(first_parents[:, :mix_children, :, :], second_parents[:, :mix_children, :, :]),
                           genetics_util.pair_breed_random(first_parents[:, mix_children:, :, :], second_parents[:, mix_children:, :, :])], 1)


def breed_data(data, selected_parents, n_children):
    data_parents = genetics_util.select_breeding_pairs_from_indicies(data, selected_parents)
    children = split_breeding(*data_parents, n_children)
    return children


def mutate_binary(data, mutation_rate):
    return cp.bitwise_xor(data, cp.random.binomial(1, mutation_rate, data.shape))


def mutate_integer(data, mutation_rate, rollover):
    return cp.mod(data + cp.random.binomial(1, mutation_rate, data.shape), rollover)

N = 8
population_size = 100
keep_best = int(0.8 * population_size)
n_children = population_size - keep_best
n_populations = 10
n_trajectories = 200
noise_prob = 0.02
mutation_rate = 0.001

init_p = 0.5
init_avg_k = 2
max_k = 8

n_generations = 2000
n_memory_timesteps = 10

input_state = make_memory_input_state(N)
input_state_batched = cp.broadcast_to(cp.expand_dims(cp.expand_dims(input_state, -2), -2), (input_state.shape[0], n_populations, population_size, input_state.shape[1]))

functions = cp.random.randint(0, 2, (n_populations, population_size, N, 1 << max_k), dtype=cp.uint8).astype(cp.bool_)
connectivity = cp.random.randint(0, N, (n_populations, population_size, N, max_k), dtype=cp.uint8)
used_connectivity = cp.random.binomial(1, init_avg_k/max_k, (n_populations, population_size, N, max_k), dtype=cp.bool_)

input_state_traj = cp.broadcast_to(cp.expand_dims(input_state_batched, 0), (n_trajectories, *input_state_batched.shape))
best_populations = [None] * n_populations
best_errors = cp.array([cp.inf] * n_populations)
for generation in range(n_generations):
    updated_states = run_dynamics_forward(input_state_traj, functions, connectivity, used_connectivity, n_memory_timesteps + np.random.randint(0, 3), noise_prob)
    error_rates = evaluate_memory_task(updated_states)
    sorted_error_rates = cp.argsort(error_rates, axis=-1)
    expand_sort_error_rates = cp.expand_dims(cp.expand_dims(sorted_error_rates, -1), -1)

    parents = genetics_util.sample_breeding_pairs_idx([n_populations], n_children)

    best_functions = cp.take_along_axis(functions, expand_sort_error_rates, 1)[:, :keep_best, :, :]
    best_connectivity = cp.take_along_axis(connectivity, expand_sort_error_rates, 1)[:, :keep_best, :, :]
    best_used_connectivity = cp.take_along_axis(used_connectivity, expand_sort_error_rates, 1)[:, :keep_best, :, :]

    function_children = mutate_binary(breed_data(best_functions, parents, n_children), mutation_rate)
    connectivity_children = mutate_integer(breed_data(best_connectivity,
                                   parents, n_children), mutation_rate, N)
    used_connectivity_children = mutate_binary(breed_data(used_connectivity,
                                   parents, n_children), mutation_rate)



    functions = cp.concatenate([best_functions, function_children], axis=1)
    connectivity = cp.concatenate([best_connectivity, connectivity_children], axis=1)
    used_connectivity = cp.concatenate([best_used_connectivity, used_connectivity_children], axis=1)

    population_errors = cp.mean(error_rates, axis=1)
    if generation % 100 == 0:
        print("GENERATION {}".format(generation))
    for i, error in enumerate(population_errors):
        if error < best_errors[i]:
            best_errors[i] = error
            best_populations[i] = (cp.asnumpy(best_functions[i]), cp.asnumpy(best_connectivity[i]), cp.asnumpy(best_used_connectivity[i]))
            print("population {} mean error rate: {}".format(i, error))


out_dir = os.path.join(os.getenv("HOME"),"boolean_network_data/memory_evolution_results/{}".format(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))
os.makedirs(out_dir, exist_ok=False)
with open(os.path.join(out_dir, 'best_populations.pk'), 'wb') as f:
    pickle.dump(best_populations, f)

