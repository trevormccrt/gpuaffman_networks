import cupy as cp
import datetime
import json
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.getenv("HOME"), "gpuaffman_networks/"))

import binary_core, limit_cycles

out_dir = os.path.join(os.getenv("HOME"), "boolean_network_data/limit_cycle_lengths/{}".format(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))
os.makedirs(out_dir, exist_ok=False)


def run_limit_cycle_experiment(batch_size, N, k, P, max_iters):
    init_states = cp.array(binary_core.random_binary_data((batch_size, N), 0.5))
    functions = cp.array(binary_core.random_binary_data((batch_size, N, 1<<k), P))
    connectivity = cp.array(np.random.randint(0, N, (batch_size, N, k)))
    results = limit_cycles.measure_limit_cycle_lengths(init_states, functions, connectivity, max_n_iter=max_iters, verbose=True)
    print("Done N: {}, k: {}".format(N, k))
    return results


batch_size = 1000
k=2
P = 0.5
N_vals =  np.floor(2 * np.logspace(start=0, stop=4, num=15)).astype(np.int64)
results = []

meta = {"k":k, "N":list([int(x) for x in N_vals]), "P": P}

with open(os.path.join(out_dir, 'meta.json'), 'w+') as f:
    json.dump(meta, f)


for N in N_vals:
    results.append(run_limit_cycle_experiment(batch_size, N, k, P, 10 * N))
    np.save(os.path.join(out_dir, "results.npy"), np.array(results, dtype=object), allow_pickle=True)

