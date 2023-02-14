import atexit
import sys
import os
sys.path.append(os.path.join(os.getenv("HOME"), "gpuaffman_networks/"))

import cupy as cp
import datetime
import numpy as np

import pickle

import cupy.cuda.device

from genetics import ragged_task_evolution


def make_memory_input_state(N):
    input_state = cp.zeros((4, N), dtype=cp.uint8)
    input_state[0, (0, 1)] = (False, False)
    input_state[1, (0, 1)] = (False, True)
    input_state[2, (0, 1)] = (True, False)
    input_state[3, (0, 1)] = (True, True)
    return input_state


def evaluate_memory_task(data):
    error_rate_0 = np.mean(np.equal(data[:, 0, :, :, 2], False), axis=0)
    error_rate_1 = np.mean(np.equal(data[:, 1, :, :, 2], False), axis=0)
    error_rate_2 = np.mean(np.equal(data[:, 2, :, :, 2], False), axis=0)
    error_rate_3 = np.mean(np.equal(data[:, 3, :, :, 2], True), axis=0)
    return np.mean(np.stack([error_rate_0, error_rate_1, error_rate_2, error_rate_3], axis=-1), axis=-1)


if __name__ == "__main__":
    os.environ["CUPY_ACCELERATORS"] = "cutensor"
    cupy.cuda.device.Device(0).use()
    N = 12
    population_size = 70
    keep_best = int(0.8 * population_size)
    n_children = population_size - keep_best
    n_populations = 10
    n_trajectories = 200
    noise_prob = 0.01
    mutation_rate = 0.001

    init_p = 0.5
    init_avg_k = 2
    max_k = 8

    n_generations = 200
    n_memory_timesteps = 15

    input_state = make_memory_input_state(N)
    input_state_batched = cp.broadcast_to(cp.expand_dims(cp.expand_dims(input_state, -2), -2), (input_state.shape[0], n_populations, population_size, input_state.shape[1]))

    functions = cp.random.randint(0, 2, (n_populations, population_size, N, 1 << max_k), dtype=cp.uint8).astype(cp.bool_)
    connectivity = cp.random.randint(0, N, (n_populations, population_size, N, max_k), dtype=cp.uint8)
    used_connectivity = cp.random.binomial(1, init_avg_k/max_k, (n_populations, population_size, N, max_k), dtype=cp.bool_)

    best_organisms = [None] * n_populations
    best_errors = np.array([np.inf] * n_populations)
    checkpoint_generations = []
    checkpoint_organisms = []
    checkpoint_errors = []

    binary_mutation_fn = lambda x: ragged_task_evolution.mutate_binary(x, mutation_rate)
    integer_mutation_fn = lambda x: ragged_task_evolution.mutate_integer(x, mutation_rate, N)


    def f_exit():
        out_dir = os.path.join(os.getenv("HOME"),"boolean_network_data/and_evolution_results/{}".format(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))
        os.makedirs(out_dir, exist_ok=False)
        with open(os.path.join(out_dir, 'best_populations.pk'), 'wb') as f:
            pickle.dump(best_organisms, f)
        with open(os.path.join(out_dir, 'checkpoint_organisms.pk'), 'wb') as f:
            pickle.dump(checkpoint_organisms, f)
        np.save(os.path.join(out_dir, "best_errors.npy"), cp.asnumpy(best_errors))
        np.save(os.path.join(out_dir, "checkpoint_errors.npy"), cp.asnumpy(checkpoint_errors))
        np.save(os.path.join(out_dir, "checkpoint_generations.npy"), cp.asnumpy(checkpoint_generations))

    atexit.register(f_exit)

    for generation in range(n_generations):
        functions, connectivity, used_connectivity, population_errors = ragged_task_evolution.evolutionary_step(
            input_state_batched, n_trajectories, functions, connectivity, used_connectivity,
            n_memory_timesteps + np.random.randint(0, 5), noise_prob, evaluate_memory_task,
            ragged_task_evolution.split_breed_data, n_children, binary_mutation_fn, integer_mutation_fn)
        if generation % 100 == 0:
            print("GENERATION {}".format(generation))
            checkpoint_generations.append(generation)
            checkpoint_organisms.append((functions[:, 0, ...], connectivity[:, 0, ...], used_connectivity[:, 0, ...]))
        for i, error in enumerate(population_errors):
            if error < best_errors[i]:
                best_errors[i] = error
                best_organisms[i] = (cp.asnumpy(functions[i, 0, ...]), cp.asnumpy(connectivity[i, 0, ...]), cp.asnumpy(used_connectivity[i, 0, ...]))
                print("population {} mean error rate: {}".format(i, error))
