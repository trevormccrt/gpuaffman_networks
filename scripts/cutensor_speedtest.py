import cupy as cp
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import time

import cuda_binary_core

function_dimension = 3
batch_sizes = 3 * np.floor(np.logspace(start=3, stop=8, num=15)).astype(np.int64)
all_times_def = []
all_times_cutensor = []
dtype = np.bool_

for i, batch_size in enumerate(batch_sizes):
    mempool = cp.get_default_memory_pool()
    print(batch_size)
    data = np.random.randint(0, 2, (batch_size, function_dimension)).astype(dtype)
    functions = np.random.randint(0, 2, (batch_size, 1<<function_dimension)).astype(dtype)
    data_cp = cp.array(data)
    functions_cp = cp.array(functions)
    if not i:
        cuda_binary_core.apply_binary_function(data_cp, functions_cp)
    os.environ["CUPY_ACCELERATORS"] = "cub"
    try:
        print("starting default")
        def_start_time = time.time()
        cuda_binary_core.apply_binary_function(data_cp, functions_cp)
        def_end_time = time.time()
        print("default time: {}".format(def_end_time - def_start_time))
    except:
        print("Default Failed")
    os.environ["CUPY_ACCELERATORS"] = "cutensor"
    try:
        print("starting cutensor")
        cut_start_time = time.time()
        cuda_binary_core.apply_binary_function(data_cp, functions_cp)
        cut_end_time = time.time()
    except:
        print("cutensor failed")
        break
    print("cutensor time: {}".format(cut_end_time - cut_start_time))
    all_times_def.append(def_end_time - def_start_time)
    all_times_cutensor.append(cut_end_time - cut_start_time)
    print("Used memory: {}".format(mempool.used_bytes()/1e9))
    print("\n")

fig, axs = plt.subplots()
axs.plot(batch_sizes[:len(all_times_def)], all_times_def, label="default")
axs.plot(batch_sizes[:len(all_times_cutensor)], all_times_cutensor, label="cutensor")
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
