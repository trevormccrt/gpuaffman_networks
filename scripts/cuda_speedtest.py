import cupy as cp
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import time

from boolean_networks import binary_core, cuda_binary_core

function_dimension = 3
batch_sizes = 3 * np.floor(np.logspace(start=3, stop=8, num=15)).astype(np.int64)
all_times_numpy = []
all_times_cupy = []
dtype = np.bool_

data_unit = binary_core.random_binary_data((1, function_dimension), 0.5).astype(dtype)
function_unit = binary_core.random_binary_data((1, 1<<function_dimension), 0.5).astype(dtype)

for i, batch_size in enumerate(batch_sizes):
    print(batch_size)
    data = np.random.randint(0, 2, (batch_size, function_dimension)).astype(dtype)#np.tile(data_unit, (batch_size, 1))
    functions = np.random.randint(0, 2, (batch_size, 1<<function_dimension)).astype(dtype)#np.tile(function_unit, (batch_size, 1))
    data_cp = cp.array(data)
    functions_cp = cp.array(functions)
    if not i:
        cuda_binary_core.apply_binary_function(data_cp, functions_cp)
    print("starting numpy")
    numpy_start_time = time.time()
    _ = binary_core.apply_binary_function(data, functions)
    numpy_end_time = time.time()
    print("numpy time: {}".format(numpy_end_time - numpy_start_time))
    print("starting cupy")
    cp_start_time = time.time()
    _ = cuda_binary_core.apply_binary_function(data_cp, functions_cp)
    cp_end_time = time.time()
    print("cupy time: {}".format(cp_end_time - cp_start_time))
    all_times_numpy.append(numpy_end_time - numpy_start_time)
    all_times_cupy.append(cp_end_time - cp_start_time)
    print("\n")

fig, axs = plt.subplots()
axs.plot(batch_sizes, all_times_numpy, label="numpy")
axs.plot(batch_sizes, all_times_cupy, label="cuda")
axs.legend()
axs.set_xscale("log")
axs.set_yscale("log")
axs.set_xlabel("Batch Size")
axs.set_ylabel("Time")
plt.show()

out_dir = os.path.join(os.getenv("HOME"), "benchmarking/{}".format(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))
os.makedirs(out_dir, exist_ok=False)
fig.savefig(os.path.join(out_dir, "perf_plot.png"))
axs.legend()
