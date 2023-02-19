import cupy as cp
import datetime
import os
import sys
sys.path.append(os.path.join(os.getenv("HOME"), "gpuaffman_networks/"))

from genetics import evolution_runners, tasks

N = 30
population_size = 70
keep_best = int(0.6 * population_size)
n_children = population_size - keep_best
n_populations = 150
n_trajectories = 3
noise_prob = 0.01
mutation_rate = 0.005

init_avg_k = 3
max_k = 4

n_generations = 600000
n_memory_timesteps = 12


out_dir = os.path.join(os.getenv("DATA_DIR"), "boolean_network_data/modular_6_evolution_results/{}".format(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))
os.makedirs(out_dir, exist_ok=False)
checkpointint_dir = os.path.join(out_dir, "checkpoint_data/")
os.makedirs(checkpointint_dir, exist_ok=False)

file_name = "batch_1.npz"

evolution_runners.evolve_batch(N, max_k, population_size, keep_best, n_populations, n_trajectories, noise_prob,
                               mutation_rate, init_avg_k, n_generations, n_memory_timesteps,
                               tasks.make_6_bit_input_state, tasks.evaluate_modular_6_task, out_dir, file_name,
                               True, checkpointing_dir=checkpointint_dir, checkpointing_freq=100)
