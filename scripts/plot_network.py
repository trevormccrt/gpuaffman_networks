import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os

import general_network_visualization

#plt.style.use('dark_background')
N = 8
k_max = 3
init_avg_k = 2.0

functions = np.random.randint(0, 2, (N, 1<<k_max)).astype(np.bool_)
connectivity = np.random.randint(0, N, (N, k_max))
used_connectivity = np.random.binomial(1, init_avg_k/k_max, (N, k_max)).astype(np.bool_)

g = general_network_visualization.influence_graph_from_ragged_spec(functions, connectivity, used_connectivity)
pos = nx.spring_layout(g)


out_dir = os.path.join("/home/trevor/boolean_network_data/plots_and_misc_material/", "example_network.png")
fig, axs = plt.subplots(figsize=(4,4))
general_network_visualization.plot_network_directed(g, pos, axs, ["C0"] * len(g.nodes), colorbar=True)
plt.savefig(out_dir, dpi=400)
#plt.show()
