import datetime
import multiprocessing as mp
import os
import random

import numpy as np

from genetics import natural_computation

out_dir = os.path.join(os.getenv("DATA_DIR"), "boolean_network_data/naturally_computing_networks/{}".format(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))
os.makedirs(out_dir, exist_ok=False)


N=40
k_max=4
avg_k=2
desired_rank=6
max_computation_time=12


def find_network_runner(random_seed, this_n_to_find):
    import numpy as np
    np.random.seed(random_seed)
    return natural_computation.find_naturally_computing_networks(N, k_max, avg_k, desired_rank, this_n_to_find, max_computation_time, batch_size=10000)


n_threads = 30
n_per_thread = 8
inputs = [(random.randint(0, int(1e9)), n_per_thread) for _ in range(n_threads)]

p = mp.Pool(processes=15)
results = p.starmap(find_network_runner, inputs)
all_fns = np.concatenate([x[0] for x in results], axis=0)
all_conn = np.concatenate([x[1] for x in results], axis=0)
all_used = np.concatenate([x[2] for x in results], axis=0)
all_tts = np.concatenate([x[3] for x in results], axis=0)
all_out_nodes = np.concatenate([x[4] for x in results], axis=0)
all_in_nodes = np.concatenate([x[5] for x in results], axis=0)
all_times = np.concatenate([x[6] for x in results], axis=0)

np.savez(os.path.join(out_dir, "data.npz"), functions=all_fns, connectivity=all_conn, used_connectivity=all_used,
         truth_tables=all_tts, output_nodes=all_out_nodes, input_nodes=all_in_nodes, computation_times=all_times, N=N, k_max=k_max, avg_k=avg_k, desired_rank=desired_rank)

print("")


