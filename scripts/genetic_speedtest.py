import cupy as cp
from cupyx.profiler import benchmark
import numpy as np


from genetics import ragged_task_evolution, tasks


def sample_parents(functions, connectivity, used_connectivity, n_populations, n_children, expand_sort_error_rates, breeding_fn):
    parents = ragged_task_evolution.sample_breeding_pairs_idx([n_populations], n_children)

    best_functions = np.take_along_axis(functions, expand_sort_error_rates, 1)[:, :keep_best, :, :]
    best_connectivity = np.take_along_axis(connectivity, expand_sort_error_rates, 1)[:, :keep_best, :, :]
    best_used_connectivity = np.take_along_axis(used_connectivity, expand_sort_error_rates, 1)[:, :keep_best, :, :]

    function_children, connectivity_children, used_connectivity_children = breeding_fn(
        ragged_task_evolution.select_breeding_pairs_from_indicies(best_functions, parents),
        ragged_task_evolution.select_breeding_pairs_from_indicies(best_connectivity, parents),
        ragged_task_evolution.select_breeding_pairs_from_indicies(best_used_connectivity, parents))
    return best_functions, best_connectivity, best_used_connectivity, function_children, connectivity_children, used_connectivity_children


def profile_evolutionary_step(input_states, n_trajectories, functions, connectivity, used_connectivity, n_timesteps, noise_prob, f_eval, breeding_fn, n_children, mutation_fn):
    n_populations = input_states.shape[-3]
    pop_size = input_states.shape[-2]
    keep_best = pop_size - n_children

    error_rate_benchmark = benchmark(ragged_task_evolution.evaluate_populations,
                                     (input_states, n_trajectories, functions, connectivity, used_connectivity, n_timesteps, noise_prob, f_eval),
                                     n_repeat=10)

    error_rate_calc_time = np.mean(error_rate_benchmark.cpu_times) + np.mean(error_rate_benchmark.gpu_times)

    error_rates = ragged_task_evolution.evaluate_populations(input_states, n_trajectories, functions, connectivity, used_connectivity, n_timesteps, noise_prob, f_eval)

    sorted_error_rates = np.argsort(error_rates, axis=-1)
    expand_sort_error_rates = np.expand_dims(np.expand_dims(sorted_error_rates, -1), -1)

    children_production_benchmark = benchmark(sample_parents, (functions, connectivity, used_connectivity, n_populations, n_children, expand_sort_error_rates, breeding_fn),
                                     n_repeat=10)

    children_production_calc_time = np.mean(children_production_benchmark.cpu_times) + np.mean(children_production_benchmark.gpu_times)

    best_functions, best_connectivity, best_used_connectivity, function_children, connectivity_children, used_connectivity_children = sample_parents(functions, connectivity, used_connectivity, n_populations, n_children, expand_sort_error_rates, breeding_fn)


    mutation_benchmark = benchmark(mutation_fn, (function_children, connectivity_children, used_connectivity_children), n_repeat=10)

    mutation_calc_time = np.mean(mutation_benchmark.cpu_times) + np.mean(
        mutation_benchmark.gpu_times)

    return error_rate_calc_time, children_production_calc_time, mutation_calc_time


population_size = 70
keep_best = int(0.4 * population_size)
n_populations = 100
n_timesteps = 10
n_traj = 15
N = 40
init_avg_k = 3
k_max = 4
noise_prob = 0.01
mutation_rate = 0.005

f_input_state = tasks.make_4_bit_input_state
f_eval = tasks.evaluate_pnas_task
f_breed = ragged_task_evolution.pair_breed_swap_all
f_mutate = lambda f, c, uc: ragged_task_evolution.mutate_equal_prob(f, c, uc, mutation_rate)


input_state = cp.array(f_input_state(N))
n_children = population_size - keep_best

input_state_batched = np.broadcast_to(np.expand_dims(np.expand_dims(input_state, -2), -2),
                                      (input_state.shape[0], n_populations, population_size, input_state.shape[1]))

functions = cp.random.randint(0, 2, (n_populations, population_size, N, 1 << k_max)).astype(
    cp.bool_)
connectivity = cp.random.randint(0, N, (n_populations, population_size, N, k_max)).astype(cp.uint8)
used_connectivity = cp.random.binomial(1, init_avg_k / k_max, (n_populations, population_size, N, k_max)).astype(cp.bool_)

error_rate_calc_time, children_production_calc_time, mutation_calc_time = profile_evolutionary_step(input_state_batched, n_traj, functions, connectivity, used_connectivity, n_timesteps, noise_prob, f_eval, f_breed, n_children, f_mutate)

print(error_rate_calc_time)
print("")
