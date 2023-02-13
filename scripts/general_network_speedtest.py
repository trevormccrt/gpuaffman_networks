import cupy as cp
import cupy.cuda.memory
from cupyx.profiler import benchmark
import matplotlib.pyplot as plt
import numpy as np
import os

import general_network

os.environ["CUPY_ACCELERATORS"] = "cutensor"

N = 100
max_k = 8
init_avg_k = 2
batch_sizes = 3 * np.floor(np.logspace(start=3, stop=5, num=15)).astype(np.int64)
all_times_np = []
all_times_cp = []
for batch_size in batch_sizes:
        print(batch_size)
        states = np.random.randint(0, 2, (batch_size, N), dtype=cp.uint8).astype(cp.bool_)
        functions = np.random.randint(0, 2, (batch_size, N, 1 << max_k), dtype=cp.uint8).astype(cp.bool_)
        connectivity = np.random.randint(0, N, (batch_size, N, max_k), dtype=cp.int16)
        times_np = benchmark(general_network.state_update, (states, functions, connectivity), n_repeat=10)
        all_times_np.append(np.mean(times_np.cpu_times) + np.mean(times_np.gpu_times))

        try:
            states_cp = cp.array(states)
            functions_cp = cp.array(functions)
            connectivity_cp = cp.array(connectivity)

            times_cp = benchmark(general_network.state_update, (states_cp, functions_cp, connectivity_cp), n_repeat=10)

            all_times_cp.append(np.mean(times_cp.cpu_times) + np.mean(times_cp.gpu_times))
        except cupy.cuda.memory.OutOfMemoryError:
            print("failed {}".format(batch_size))


fig, axs = plt.subplots()
axs.plot(batch_sizes[:len(all_times_np)], all_times_np, label="CPU")
axs.plot(batch_sizes[:len(all_times_cp)], all_times_cp, label="cuda")
axs.legend()
axs.set_xscale("log")
axs.set_yscale("log")
axs.set_xlabel("Batch Size")
axs.set_ylabel("Time")
plt.show()
