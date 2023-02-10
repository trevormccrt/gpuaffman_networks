import cupy as cp

import ragged_general_network





def run_generation(functions, connectivity, active_connections, evaluation_fn, breeding_fn, mutation_fn):
    next_generation = ragged_general_network.ragged_k_state_update(states, functions, connectivity, active_connections)
