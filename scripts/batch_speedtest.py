import cupy as cp
from cupyx.profiler import benchmark
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import time

os.environ["CUPY_ACCELERATORS"] = "cutensor"


import cuda_binary_core

function_dimension = 3
batch_sizes = 3 * np.floor(np.logspace(start=3, stop=8, num=15)).astype(np.int64)
n_batch_dims = [1, 2, 3, 4]
all_cpu_times = np.zeros((len(batch_sizes), len(n_batch_dims)))
all_gpu_times = np.zeros((len(batch_sizes), len(n_batch_dims)))
dtype = np.bool_

for i, batch_size in enumerate(batch_sizes):
    for j, n_dims in enumerate(n_batch_dims):
        batch_dim = np.floor(np.power(batch_size, 1/n_dims)).astype(np.int64)
        batch_shape = tuple([batch_dim] * n_dims)
        print("{}, {}".format(batch_size, batch_shape))
        data = np.random.randint(0, 2, (*batch_shape, function_dimension)).astype(dtype)
        functions = np.random.randint(0, 2, (*batch_shape, 1<<function_dimension)).astype(dtype)
        data_cp = cp.array(data)
        functions_cp = cp.array(functions)
        try:
            times = benchmark(cuda_binary_core.apply_binary_function, (data_cp, functions_cp), n_repeat=10)
            all_cpu_times[i, j] = np.mean(times.cpu_times)
            all_gpu_times[i, j] = np.mean(times.gpu_times)
            print("")
        except:
            print("{} failed".format(batch_shape))


fig, axs = plt.subplots(ncols=2, sharey=True)
for i, batch_dim in enumerate(n_batch_dims):
     axs[0].plot(batch_sizes, all_cpu_times[:, i], color="C{}".format(i), label="{}".format(batch_dim))
     axs[1].plot(batch_sizes, all_gpu_times[:, i], color="C{}".format(i))



axs[0].legend()
axs[0].set_xscale("log")
axs[1].set_xscale("log")
axs[0].set_yscale("log")
axs[1].set_yscale("log")
axs[0].set_xlabel("Batch Size")
axs[1].set_xlabel("Batch Size")
axs[0].set_ylabel("Time")
axs[0].set_title("CPU Time")
axs[1].set_title("GPU Time")
plt.show()

out_dir = os.path.join(os.getenv("HOME"), "benchmarking/{}".format(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))
os.makedirs(out_dir, exist_ok=False)
fig.savefig(os.path.join(out_dir, "perf_plot.png"))
axs.legend()
