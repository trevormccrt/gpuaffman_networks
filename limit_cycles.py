import numpy as np
import cupy as cp

import general_network, ragged_general_network


def measure_limit_cycle_lengths(init_state, functions, connections, used_connections=None, max_n_iter=1000, verbose=False, dry_term=1000):
    all_states = np.expand_dims(init_state, 0)
    cycle_lengths = np.zeros(np.shape(init_state)[-2], dtype=np.int64)
    cycle_start_end = np.zeros((np.shape(init_state)[-2], 2), dtype=np.int64)
    found_cycles = [None] * int(init_state.shape[0])
    connections = np.broadcast_to(connections, (*init_state.shape, connections.shape[-1]))
    used_connections = np.broadcast_to(used_connections, (*init_state.shape, used_connections.shape[-1]))
    functions = np.broadcast_to(functions, (*init_state.shape, functions.shape[-1]))
    still_evolving = np.arange(start=0, stop=np.shape(init_state)[-2], step=1)
    last_found = 0
    try:
        for i in range(max_n_iter):
            if i - last_found > dry_term:
                print("NO NEW CYCLES FOUND, EXITING")
                break
            if used_connections is None:
                new_state = general_network.state_update(all_states[i, :, :], functions[still_evolving, :, :], connections[still_evolving, :, :])
            else:
                new_state = ragged_general_network.ragged_k_state_update(all_states[i, :, :], functions[still_evolving, :, :],
                                                         connections[still_evolving, :, :], used_connections[still_evolving, :, :])
            equal_indicies = np.argwhere(np.all(np.equal(new_state, all_states[:i + 1, :, :]), -1))
            if isinstance(equal_indicies, cp.ndarray):
                equal_indicies = cp.asnumpy(equal_indicies)
            this_indicies = np.arange(start=0, stop=np.shape(new_state)[0], step=1)
            if np.size(equal_indicies):
                last_found = i
                for match in equal_indicies:
                    match_idx = still_evolving[match[1]]
                    cycle_lengths[match_idx] = i - match[0] + 1
                    cycle_start_end[match_idx] = np.array([match[0], i+1])
                    found_cycles[match_idx] = all_states[match[0]:i + 1, match[1], :]
            all_match_idx = [x[1] for x in equal_indicies]
            this_keep_indices = np.delete(this_indicies, all_match_idx)
            still_evolving = np.delete(still_evolving, all_match_idx)
            all_states = np.concatenate([all_states[:, this_keep_indices, :], np.expand_dims(new_state[this_keep_indices, :], 0)], axis=0)
            if verbose:
                print("{}: {}, {}, {}".format(i, len(still_evolving), init_state.shape, i - last_found))
            if np.all(cycle_lengths):
                break
    except cp.cuda.memory.OutOfMemoryError:
        print("OUT OF MEMORY. EXITING")
    return cycle_lengths, cycle_start_end, len(still_evolving), i, found_cycles
