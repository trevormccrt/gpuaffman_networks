import cupy as cp
import datetime
import numpy as np
import os

from genetics import ragged_task_evolution, evolution_runners, natural_computation

input_data_path = os.path.join(os.getenv("DATA_DIR"), "boolean_network_data/naturally_computing_networks/2023-03-08-15-26-26/data.npz")
data = np.load(input_data_path)

to_use_start = 0
to_use_end = 100

functions = data["functions"][to_use_start:to_use_end]
connectivity = data["connectivity"][to_use_start:to_use_end]
used_connectivity = data["used_connectivity"][to_use_start:to_use_end]
desired_truth_tables = cp.array(data["truth_tables"][to_use_start:to_use_end])
output_nodes = data["output_nodes"][to_use_start:to_use_end]

population_size = 70

functions = cp.array(np.tile(np.expand_dims(functions, 1), (1, population_size, 1, 1)))
connectivity = cp.array(np.tile(np.expand_dims(connectivity, 1), (1, population_size, 1, 1)))
used_connectivity = cp.array(np.tile(np.expand_dims(used_connectivity, 1), (1, population_size, 1, 1)))

keep_best = int(0.4 * population_size)
n_children = population_size - keep_best

n_trajectories = 5
noise_prob = 0.01
mutation_rate = 0.005

n_generations = 1000000
n_memory_timesteps = np.max(data["computation_times"][to_use_start: to_use_end])

a = os.getenv("DATA_DIR")
out_dir = os.path.join(a, "boolean_network_data/natural_evolution_results/{}".format(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))
os.makedirs(out_dir, exist_ok=False)
checkpointint_dir = os.path.join(out_dir, "checkpoint_data/")
os.makedirs(checkpointint_dir, exist_ok=False)


file_name = "batch_1.npz"

f_breed = ragged_task_evolution.pair_breed_swap_all

f_eval = lambda x: natural_computation.eval_natural_task(x, output_nodes, desired_truth_tables)

input_state = natural_computation.make_input_state(np.arange(start=0, stop=data["desired_rank"], step=1), data["N"])

input_state_batched = cp.array(np.broadcast_to(np.expand_dims(np.expand_dims(input_state, -2), -2),
                                          (input_state.shape[0], functions.shape[0], population_size, input_state.shape[1])))

f_mutate = lambda f, c, uc: ragged_task_evolution.mutate_equal_prob(f, c, uc, mutation_rate)
evolution_runners.evolve_from_spec(input_state_batched, functions, connectivity, used_connectivity, keep_best,
                                   n_trajectories, noise_prob, n_generations, n_memory_timesteps, n_children,
                                   f_eval, f_breed, f_mutate, out_dir, file_name,
                                   using_cuda=True, checkpointing_dir=checkpointint_dir, checkpointing_freq=500)
