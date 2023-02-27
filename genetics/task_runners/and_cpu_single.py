import datetime
import os
import sys
sys.path.append(os.path.join(os.getenv("HOME"), "gpuaffman_networks/"))

from genetics import evolution_runners, tasks, ragged_task_evolution

N = 8
population_size = 70
keep_best = int(0.8 * population_size)
n_children = population_size - keep_best
n_populations = 40
n_trajectories = 200
noise_prob = 0.01
mutation_rate = 0.001

init_avg_k = 2
max_k = 2

n_generations = 200
n_memory_timesteps = 10

a = os.getenv("DATA_DIR")
out_dir = os.path.join(a, "boolean_network_data/and_evolution_results/{}".format(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))
os.makedirs(out_dir, exist_ok=False)


file_name = "batch_1.npz"

evolution_runners.evolve_batch(N, max_k, population_size, keep_best, n_populations, n_trajectories, noise_prob,
                               mutation_rate, init_avg_k, n_generations, n_memory_timesteps,
                               tasks.make_2_bit_input_state, tasks.evaluate_and_task, f_breed, ragged_task_evolution.mutate_equal_prob, out_dir, file_name, using_cuda=False)
