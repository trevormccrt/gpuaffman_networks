
from cupyx.profiler import benchmark
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp

import binary_core, general_network, linear_network

N = 20

batch_sizes = 3 * np.floor(np.logspace(start=3, stop=7, num=15)).astype(np.int64)
all_times_general = []
all_times_linear = []
for batch_size in batch_sizes:
    print(batch_size)
    try:
        functions = cp.array(np.random.randint(0, 1, (batch_size, N, 8)))
        indexes = np.arange(start=0, stop=N, step=1)
        connections = cp.array(np.broadcast_to(np.expand_dims(np.stack([np.roll(indexes, 1), indexes, np.roll(indexes, -1)], axis=-1), 0), (batch_size, N, 3)))
        states = cp.array(binary_core.random_binary_data((batch_size, N), 0.5))
    except:
        print("failed to create data")
        break

    try:
        time_general = benchmark(general_network.state_update, (states, functions, connections), n_repeat=10)
        all_times_general.append(np.mean(time_general.gpu_times) + np.mean(time_general.cpu_times))
    except:
        print("general failed")
    try:
        time_linear = benchmark(linear_network.state_update, (states, functions), n_repeat=10)
        all_times_linear.append(np.mean(time_linear.gpu_times) + np.mean(time_linear.cpu_times))
    except:
        print("linear failed")

fig, axs = plt.subplots()
axs.plot(batch_sizes[:len(all_times_general)], all_times_general, label="General")
axs.plot(batch_sizes[:len(all_times_linear)], all_times_linear, label="Linear")
axs.legend()
axs.set_xscale("log")
axs.set_yscale("log")
axs.set_xlabel("Batch Size")
axs.set_ylabel("Time")
plt.show()


