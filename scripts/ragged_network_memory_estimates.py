import matplotlib.pyplot as plt
import numpy as np

def estimate_memory_general(batch, N, k):
    state_usage = batch * N
    fns_usage = batch * N * 1<<k
    conn_usage = batch * N * k * 2
    used = batch * N * k
    return state_usage + fns_usage + conn_usage + used


batch_sizes = 1 * np.floor(np.logspace(start=3, stop=7, num=15)).astype(np.int64)
N = 20
k = 8
a = estimate_memory_general(batch_sizes, N, 8)/1e9

fig, axs = plt.subplots()
axs.set_xscale("log")
axs.set_yscale("log")
axs.plot(batch_sizes, a, label="predicted")
axs.plot(batch_sizes, np.ones(len(batch_sizes)) * 24, linestyle="--", color="black", label="3090 Limit")
axs.set_title("General Network State Update Memory Usage, \n N={}, k={}".format(N, k))
axs.set_xlabel("Batch Size")
axs.set_ylabel("Memory Usage (GB)")
axs.legend()
plt.show()

print("")
