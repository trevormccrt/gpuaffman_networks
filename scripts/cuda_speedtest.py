import cupy as cp
from cupyx.profiler import benchmark
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os


import binary_core

os.environ["CUPY_ACCELERATORS"] = "cutensor"

function_dimension = 8

batch_sizes = 1 * np.floor(np.logspace(start=3, stop=8, num=15)).astype(np.int64)

all_times_numpy = []
all_times_cupy = []
dtype = np.bool_


data_unit = np.random.randint(0, 2, (1, function_dimension)).astype(dtype)
function_unit = np.random.randint(0, 2, (1, 1<<function_dimension)).astype(dtype)
for i, batch_size in enumerate(batch_sizes):
    print(batch_size)
    try:
        data = np.tile(data_unit, (batch_size, 1))
        functions = np.tile(function_unit, (batch_size, 1))
    except:
        print("batch generation failed")
        break
    print("starting numpy")
    try:
        times = benchmark(binary_core.apply_binary_function, (data, functions), n_repeat=10)
        np_time = np.mean(times.cpu_times) + np.mean(times.gpu_times)
        print("numpy time: {}".format(np_time))
        all_times_numpy.append(np_time)
    except:
        print("numpy failed")
    print("starting cupy")
    try:
        data_cp = cp.array(data)
        functions_cp = cp.array(functions)
        times = benchmark(binary_core.apply_binary_function, (data_cp, functions_cp), n_repeat=10)
        cp_time = np.mean(times.cpu_times) + np.mean(times.gpu_times)
        print("cupy time: {}".format(cp_time))
        all_times_cupy.append(cp_time)
    except:
        print("cupy failed")


    print("\n")

fig, axs = plt.subplots()
axs.plot(batch_sizes[:len(all_times_numpy)], batch_sizes[:len(all_times_numpy)]/all_times_numpy, label="Ryzen 9 5950X (Single Core)")
axs.plot(batch_sizes[:len(all_times_cupy)], batch_sizes[:len(all_times_cupy)]/all_times_cupy, label="RTX 3090")
axs.legend()
axs.set_xscale("log")
axs.set_yscale("log")
axs.set_xlabel("Batch Size")
axs.set_ylabel("Update Rate [Hz]")
axs.set_title("k={}".format(function_dimension))
plt.show()

out_dir = os.path.join(os.getenv("HOME"), "benchmarking/{}".format(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))
os.makedirs(out_dir, exist_ok=False)
fig.savefig(os.path.join(out_dir, "perf_plot.png"))
axs.legend()
