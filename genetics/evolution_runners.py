import numpy as np
import os

from genetics import ragged_task_evolution


def dump_population_data(out_dir, file_name, functions, connectivity, used_connectivity, errors, N, max_k,
                         noise_prob, mutation_rate, timesteps, death_ratio, population_size, n_trajectories, timestamp):
    np.savez(os.path.join(out_dir, file_name),
             functions=functions, connectivity=connectivity, used_connectivity=used_connectivity,
             errors=errors, N=N, max_k=max_k, noise_prob=noise_prob,
             mutation_rate=mutation_rate, timesteps=timesteps, death_ratio=death_ratio,
             population_size=population_size, n_trajectories=n_trajectories, timestamp=timestamp)


def evolve_batch(N, k_max, population_size, keep_best, n_populations, n_trajectories, noise_prob,
                 mutation_rate, init_avg_k, n_generations, timesteps, f_input_state, f_eval,
                 results_dir, results_fname, using_cuda=False, checkpointing_dir=None, checkpointing_freq=100):
    xp = np
    if using_cuda:
        import cupy
        xp = cupy
    input_state = xp.array(f_input_state(N))
    n_children = population_size - keep_best

    input_state_batched = np.broadcast_to(np.expand_dims(np.expand_dims(input_state, -2), -2),
                                          (input_state.shape[0], n_populations, population_size, input_state.shape[1]))

    functions = xp.random.randint(0, 2, (n_populations, population_size, N, 1 << k_max)).astype(
        xp.bool_)
    connectivity = xp.random.randint(0, N, (n_populations, population_size, N, k_max)).astype(xp.uint8)
    used_connectivity = xp.random.binomial(1, init_avg_k / k_max, (n_populations, population_size, N, k_max)).astype(xp.bool_)
    best_functions = np.zeros(functions[:, 0, :, :].shape).astype(np.bool_)
    best_connectivity = np.zeros(connectivity[:, 0, :, :].shape).astype(np.uint8)
    best_used_connectivity = np.zeros(used_connectivity[:, 0, :, :].shape).astype(np.bool_)
    best_errors = np.array([np.inf] * n_populations)

    binary_mutation_fn = lambda x: ragged_task_evolution.mutate_binary(x, mutation_rate)
    integer_mutation_fn = lambda x: ragged_task_evolution.mutate_integer(x, mutation_rate, N)

    for generation in range(n_generations):
        functions, connectivity, used_connectivity, population_errors = ragged_task_evolution.evolutionary_step(
            input_state_batched, n_trajectories, functions, connectivity, used_connectivity,
            timesteps + np.random.randint(0, 5), noise_prob, f_eval,
            ragged_task_evolution.split_breed_data, n_children, binary_mutation_fn, integer_mutation_fn)
        if generation % checkpointing_freq == 0:
            print("GENERATION {} ERRORS {}".format(generation, sorted(best_errors)))
            if checkpointing_dir is not None:
                dump_population_data(checkpointing_dir, "checkpoint_{}.npz".format(generation),
                                     best_functions, best_connectivity,
                                     best_used_connectivity, best_errors, N, k_max, noise_prob, mutation_rate,
                                     timesteps, keep_best, population_size, n_trajectories, generation)
        for i, error in enumerate(population_errors):
            if error < best_errors[i]:
                this_best_functions = functions[i, 0, ...]
                this_best_connectivity = connectivity[i, 0, ...]
                this_best_used_connectivity = used_connectivity[i, 0, ...]
                if using_cuda:
                    error = xp.asnumpy(error)
                    this_best_functions = xp.asnumpy(this_best_functions)
                    this_best_connectivity = xp.asnumpy(this_best_connectivity)
                    this_best_used_connectivity = xp.asnumpy(this_best_used_connectivity)
                best_errors[i] = error
                best_functions[i] = this_best_functions
                best_connectivity[i] = this_best_connectivity
                best_used_connectivity[i] = this_best_used_connectivity
                dump_population_data(results_dir, results_fname, best_functions, best_connectivity,
                                     best_used_connectivity, best_errors, N, k_max, noise_prob, mutation_rate,
                                     timesteps, keep_best, population_size, n_trajectories, generation)
