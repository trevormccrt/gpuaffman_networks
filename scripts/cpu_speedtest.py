import matplotlib.pyplot as plt
import numpy as np
import timeit

import ragged_general_network
population_size = 70
batch_sizes = np.linspace(start=0, stop=1000, num=50).astype(np.int64)#np.logspace(start=0, stop=4, num=20).astype(np.int64)
N = 8
max_k = 3
avg_k = 2.0

times = []
for i, batch_size in enumerate(batch_sizes):
    print(batch_size)
    states = np.random.binomial(1, 0.5, (batch_size, population_size, N)).astype(np.bool_)
    functions = np.random.binomial(1, 0.5, (batch_size, population_size, N, 1<<max_k)).astype(np.bool_)
    connectivity = np.random.randint(0, N, (batch_size, population_size, N, max_k)).astype(np.uint8)
    used_connectivity = np.random.binomial(1, avg_k/max_k, (batch_size, population_size, N, max_k)).astype(np.bool_)
    times.append(timeit.timeit(lambda: ragged_general_network.ragged_k_state_update(states, functions, connectivity, used_connectivity), number=20))
times = np.array(times)
normed_times = times/batch_sizes

fig, axs = plt.subplots()
axs.plot(batch_sizes, normed_times)
plt.show()
