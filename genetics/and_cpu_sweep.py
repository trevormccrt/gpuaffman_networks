import datetime
import multiprocessing as mp
import numpy as np
import os

import ragged_task_evolution
from and_task_evolution import make_and_input_state, evaluate_and_task


def run_task(N, max_k, init_avg_k, init_P, noise_prob, mutation_rate, n_generations, n_trajectories, timesteps, population_size, death_ratio, out_dir, process_id):
    np.random.seed(process_id)
    n_populations = 1
    population_size = 70
    keep_best = int(0.8 * population_size)
    n_children = population_size - keep_best

    input_state = make_and_input_state(N)
    input_state_batched = np.broadcast_to(np.expand_dims(np.expand_dims(input_state, -2), -2),
                                          (input_state.shape[0], n_populations, population_size, input_state.shape[1]))

    functions = np.random.randint(0, 2, (n_populations, population_size, N, 1 << max_k), dtype=np.uint8).astype(
        np.bool_)
    connectivity = np.random.randint(0, N, (n_populations, population_size, N, max_k), dtype=np.uint8)
    used_connectivity = np.random.binomial(1, init_avg_k / max_k, (n_populations, population_size, N, max_k)).astype(np.bool_)

    best_functions = np.copy(functions)[:, 0, :, :]
    best_connectivity = np.copy(connectivity)[:, 0, :, :]
    best_used_connectivity = np.copy(used_connectivity)[:, 0, :, :]
    best_errors = np.array([np.inf] * n_populations)

    def f_save():
        np.savez(os.path.join(out_dir, "data_{}.npz".format(process_id)),
                 functions=best_functions, connectivity=best_connectivity, used_connectivity=best_used_connectivity,
                 errors=best_errors, N=N, max_k=max_k, init_P=init_P, noise_prob=noise_prob,
                 mutation_rate=mutation_rate, timesteps=timesteps, death_ratio=death_ratio,
                 population_size=population_size, n_trajectories=n_trajectories)

    binary_mutation_fn = lambda x: ragged_task_evolution.mutate_binary(x, mutation_rate)
    integer_mutation_fn = lambda x: ragged_task_evolution.mutate_integer(x, mutation_rate, N)

    for generation in range(n_generations):
        functions, connectivity, used_connectivity, population_errors = ragged_task_evolution.evolutionary_step(
            input_state_batched, n_trajectories, functions, connectivity, used_connectivity,
            timesteps + np.random.randint(0, 5), noise_prob, evaluate_and_task,
            ragged_task_evolution.split_breed_data, n_children, binary_mutation_fn, integer_mutation_fn)
        if generation % 100 == 0:
            print("GENERATION {} ERRORS {}".format(generation, sorted(best_errors)))
            f_save()
        for i, error in enumerate(population_errors):
            if error < best_errors[i]:
                best_errors[i] = error
                best_functions[i] = functions[i, 0, ...]
                best_connectivity[i] = connectivity[i, 0, ...]
                best_used_connectivity[i] = used_connectivity[i, 0, ...]
    f_save()




out_dir = os.path.join(os.getenv("HOME"),"boolean_network_data/cpu_and_evolution_results/{}".format(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))
os.makedirs(out_dir, exist_ok=False)

n_trials = 1


n_vals = np.random.randint(4, 20, n_trials)
max_k_vals = np.random.randint(2, 5, n_trials)

inputs = []
for i, (n, k) in enumerate(zip(n_vals, max_k_vals)):
    inputs.append((n, k, 2, 0.5, 0.01, 0.001, 400, 100, 7, 70, 0.6, out_dir, i))

p = mp.Pool()
p.starmap(run_task, inputs)
