import cupy as cp
import datetime
import os
import sys
sys.path.append(os.path.join(os.getenv("HOME"), "gpuaffman_networks/"))

from genetics import evolution_runners, tasks, ragged_task_evolution

N = 15
population_size = 70
keep_best = int(0.4 * population_size)
n_children = population_size - keep_best
n_populations = 70
n_trajectories = 20
noise_prob = 0.01
mutation_rate = 0.005

init_avg_k = 2
max_k = 3

n_generations = 300000
n_memory_timesteps = 10


out_dir = os.path.join(os.getenv("DATA_DIR"), "boolean_network_data/sequential_and_evolution_results/func_noise_{}".format(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))
os.makedirs(out_dir, exist_ok=False)
checkpointint_dir = os.path.join(out_dir, "checkpoint_data/")
os.makedirs(checkpointint_dir, exist_ok=False)

file_name = "batch_1.npz"

f_mutate = lambda f, c, uc: ragged_task_evolution.mutate_equal_prob(f, c, uc, mutation_rate)

evolution_runners.evolve_random_batch(N, max_k, population_size, keep_best, n_populations, n_trajectories, noise_prob,
                                        init_avg_k, n_generations, n_memory_timesteps,
                                      tasks.make_3_bit_input_state, tasks.evaluate_sequential_and_task, ragged_task_evolution.pair_breed_swap_all, f_mutate, out_dir, file_name,
                                      True, checkpointing_dir=checkpointint_dir, checkpointing_freq=500, dynamcis_fn=ragged_task_evolution.dynamics_with_function_noise)
