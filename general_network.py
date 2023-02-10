import cupy as cnp
import cupy.cuda.memory
import numpy as np


import cuda_binary_core


def get_update_inputs(states, connections):
    connections = cnp.broadcast_to(connections, (*states.shape, connections.shape[-1]))
    return cnp.take_along_axis(cnp.expand_dims(states, -1), connections, -2)


def state_update(states, functions, connections):
    return cuda_binary_core.apply_binary_function(get_update_inputs(states, connections), functions)


def measure_limit_cycle_lengths(init_state, functions, connections, max_n_iter=1000, verbose=False, dry_term=1000):
    mempool = cnp.get_default_memory_pool()
    all_states = cnp.expand_dims(init_state, 0)
    cycle_lengths = np.zeros(cnp.shape(init_state)[-2], dtype=np.int64)
    cycles = np.zeros((cnp.shape(init_state)[-2], 2), dtype=np.int64)
    connections = cnp.broadcast_to(connections, (*init_state.shape, connections.shape[-1]))
    functions = cnp.broadcast_to(functions, (*init_state.shape, functions.shape[-1]))
    still_evolving = np.arange(start=0, stop=np.shape(init_state)[-2], step=1)
    last_found = 0
    try:
        for i in range(max_n_iter):
            if i - last_found > dry_term:
                print("NO NEW CYCLES FOUND, EXITING")
                break
            new_state = state_update(all_states[i, :, :], functions[still_evolving, :, :], connections[still_evolving, :, :])
            equal_indicies = cnp.argwhere(cnp.all(cnp.equal(new_state, all_states[:i + 1, :, :]), -1))
            equal_indicies = cnp.asnumpy(equal_indicies)
            this_indicies = np.arange(start=0, stop=np.shape(new_state)[0], step=1)
            if np.size(equal_indicies):
                last_found = i
                for match in equal_indicies:
                    match_idx = still_evolving[match[1]]
                    cycle_lengths[match_idx] = i - match[0] + 1
                    cycles[match_idx] = np.array([match[0], i+1])
            all_match_idx = [x[1] for x in equal_indicies]
            this_keep_indices = np.delete(this_indicies, all_match_idx)
            still_evolving = np.delete(still_evolving, all_match_idx)
            all_states = cnp.concatenate([all_states[:, this_keep_indices, :], cnp.expand_dims(new_state[this_keep_indices, :], 0)], axis=0)
            if verbose:
                print("{}: {}, {}, {}, {}".format(i, len(still_evolving), init_state.shape, mempool.used_bytes()/1e9, i - last_found))
            if np.all(cycle_lengths):
                break
    except cupy.cuda.memory.OutOfMemoryError:
        print("OUT OF MEMORY. EXITING")
    return cycle_lengths, cycles, len(still_evolving), i
