import cupy as cp
import cupy.cuda.memory
from cupyx.profiler import benchmark
import matplotlib.pyplot as plt
import numpy as np
import os

import ragged_general_network, general_network

os.environ["CUPY_ACCELERATORS"] = "cutensor"

function_dimension = 3
N = 8
max_k = 8
init_avg_k = 2
batch_sizes = 3 * np.floor(np.logspace(start=3, stop=8, num=15)).astype(np.int64)
all_times = []
for batch_size in batch_sizes:
    try:
        states = cp.random.randint(0, 2, (batch_size, N), dtype=cp.uint8).astype(cp.bool_)
        functions = cp.random.randint(0, 2, (batch_size, N, 1 << max_k), dtype=cp.uint8).astype(cp.bool_)
        connectivity = cp.random.randint(0, N, (batch_size, N, max_k), dtype=cp.uint8)
        #used_connectivity = cp.random.binomial(1, init_avg_k/max_k, (batch_size, N, max_k), dtype=cp.bool_)

        #times = benchmark(ragged_general_network.ragged_k_state_update, (states, functions, connectivity, used_connectivity), n_repeat=10)
        times = benchmark(general_network.state_update, (states, functions, connectivity), n_repeat=10)
        all_times.append(np.mean(times.cpu_times) + np.mean(times.gpu_times))
    except cupy.cuda.memory.OutOfMemoryError:
        print("failed {}".format(batch_size))


fig, axs = plt.subplots()
axs.plot(batch_sizes[:len(all_times)], all_times)
axs.set_xscale("log")
axs.set_yscale("log")
axs.set_xlabel("Batch Size")
axs.set_ylabel("Time")
plt.show()
