import cupy as cp
import cupy.cuda.memory
from cupyx.profiler import benchmark
from gpuaffman_networks import ragged_general_network
import matplotlib.pyplot as plt
import numpy as np

N = 25
max_k = 4
init_avg_k = 3
batch_sizes = 2 * np.floor(np.logspace(start=3, stop=6, num=15)).astype(np.int64)
all_times = []
for batch_size in batch_sizes:
    try:
        times = benchmark(ragged_general_network.ragged_k_state_update,
                          (cp.random.randint(0, 2, (batch_size, N), dtype=cp.uint8).astype(cp.bool_),
                           cp.random.randint(0, 2, (batch_size, N, 1 << max_k), dtype=cp.uint8).astype(cp.bool_),
                           cp.random.randint(0, N, (batch_size, N, max_k), dtype=cp.uint8),
                           cp.random.binomial(1, init_avg_k/max_k, (batch_size, N, max_k), dtype=cp.bool_)), n_repeat=1)
        all_times.append(np.mean(times.cpu_times) + np.mean(times.gpu_times))
    except cupy.cuda.memory.OutOfMemoryError:
        print("failed {}".format(batch_size))


fig, axs = plt.subplots()
axs.plot(batch_sizes[:len(all_times)], all_times)
axs.set_xscale("log")
axs.set_yscale("log")
axs.set_xlabel("Batch Size")
axs.set_ylabel("Network Update Time")
axs.set_title("N: {}, k: {}".format(N, max_k))
plt.show()
