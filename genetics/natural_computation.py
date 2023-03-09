import numpy as np

import binary_core, limit_cycles
from genetics import analysis_util


def make_input_state(input_bits, N):
    n_bits = len(input_bits)
    truth_table = binary_core.truth_table_columns(n_bits)
    input_state = np.zeros((1 << n_bits, N), dtype=np.uint8)
    input_state[:, input_bits] = truth_table
    return input_state


def natural_computation_search(functions, connectivity, used_connectivity, input_bits):
    N = functions.shape[-2]
    batch_size = functions.shape[0]
    raw_input_state = make_input_state(input_bits, N)
    input_state = np.broadcast_to(raw_input_state, (batch_size, *raw_input_state.shape))
    input_state = np.moveaxis(input_state, -2, 0)
    functions = np.broadcast_to(functions, (*input_state.shape, functions.shape[-1]))
    connectivity = np.broadcast_to(connectivity, (*input_state.shape, connectivity.shape[-1]))
    used_connectivity = np.broadcast_to(used_connectivity, (*input_state.shape, used_connectivity.shape[-1]))
    input_state_flat = np.reshape(input_state, (-1, input_state.shape[-1]))
    functions_flat = np.reshape(functions, (-1, *functions.shape[-2:]))
    connectivity_flat = np.reshape(connectivity, (-1, *connectivity.shape[-2:]))
    used_connectivity_flat = np.reshape(used_connectivity, (-1, *used_connectivity.shape[-2:]))
    cycle_lengths, cycle_start_end, num_not_found, _, found_cycles = limit_cycles.measure_limit_cycle_lengths(input_state_flat, functions_flat, connectivity_flat, used_connectivity_flat)
    found_cycles = np.reshape(np.array(found_cycles), input_state.shape[:-1]).T
    cycle_start_end = np.reshape(cycle_start_end, (*input_state.shape[:-1], 2))
    cycle_start_end = np.swapaxes(cycle_start_end, 0, 1)
    output_nodes = []
    effective_truth_tables = []
    computation_stablilize_time = []
    non_empty_idx = []
    for i, (all_input_results, start_end) in enumerate(zip(found_cycles, cycle_start_end)):
        try:
            frozen_nodes = np.array([np.all(x == x[0], axis=0) for x in all_input_results])
            frozen_vales = np.array([x[0] for x in all_input_results])
            frozen_always = np.all(frozen_nodes, axis=0)
            doing_computation = np.bitwise_not(np.all(frozen_vales == frozen_vales[0], axis=0))
            comp_idx = np.argwhere(np.bitwise_and(doing_computation, frozen_always)).flatten()
            effective_truth_table = frozen_vales[:, comp_idx]
            output_nodes.append(comp_idx)
            effective_truth_tables.append(effective_truth_table)
            computation_stablilize_time.append(np.max(start_end[:, 0]))
            if len(comp_idx):
                non_empty_idx.append(i)
        except TypeError:
            output_nodes.append([])
            effective_truth_tables.append([])
            computation_stablilize_time.append(0)
    return output_nodes, effective_truth_tables, computation_stablilize_time, non_empty_idx


def order_natural_computations_by_rank(output_nodes, effective_truth_tables, computation_stabilize_time, computing_idx, input_bits):
    computing_tts = [effective_truth_tables[x] for x in computing_idx]
    computing_output_nodes = [output_nodes[x] for x in computing_idx]
    computing_times = [computation_stabilize_time[x] for x in computing_idx]
    bits_used = [[analysis_util.compute_influence(y) > 0 for y in x.T] for x in computing_tts]
    active_input_bits = [[input_bits[np.argwhere(y).flatten()] for y in x] for x in bits_used]
    ranks = [[np.sum(y) for y in x] for x in bits_used]
    rank_order = np.argsort([np.mean(x) for x in ranks])[::-1]
    ordered_ranks = [ranks[x] for x in rank_order]
    ordered_tts = [computing_tts[x] for x in rank_order]
    ordered_out_nodes = [computing_output_nodes[x] for x in rank_order]
    ordered_times = [computing_times[x] for x in rank_order]
    ordered_idx = [computing_idx[x] for x in rank_order]
    ordered_in_nodes = [active_input_bits[x] for x in rank_order]
    return ordered_ranks, ordered_tts, ordered_out_nodes, ordered_in_nodes, ordered_times, ordered_idx


def find_naturally_computing_networks(N, k_max, avg_k, desired_rank, n_to_find, max_computation_time, batch_size=10000):
    found_orgs = []
    while len(found_orgs) < n_to_find:
        input_bits = np.arange(start=0, stop=desired_rank, step=1)
        functions = np.random.binomial(1, 0.5, (batch_size, N, 1 << k_max)).astype(np.bool_)
        connectivity = np.random.randint(0, N, (batch_size, N, k_max)).astype(np.uint8)
        used_connectivity = np.random.binomial(1, avg_k / k_max, (batch_size, N, k_max)).astype(np.bool_)
        output_nodes, effective_truth_tables, computation_stablilize_time, non_empty_idx = natural_computation_search(functions, connectivity, used_connectivity, input_bits)
        ordered_ranks, ordered_tts, ordered_out_nodes, ordered_in_nodes, ordered_times, ordered_idx = order_natural_computations_by_rank(output_nodes, effective_truth_tables, computation_stablilize_time, non_empty_idx, input_bits)
        for ranks, tts, out_nodes, in_nodes, time, orig_idx in zip(ordered_ranks, ordered_tts, ordered_out_nodes, ordered_in_nodes, ordered_times, ordered_idx):
            if time < max_computation_time:
                for rank, out_n, in_n, tt in zip(ranks, out_nodes, in_nodes, tts.T):
                    if rank == desired_rank:
                        found_orgs.append((functions[orig_idx], connectivity[orig_idx], used_connectivity[orig_idx], tt, out_n, in_n, time))
                        print('found one!')
                        break
    all_functions = np.stack([x[0] for x in found_orgs], axis=0)
    all_conn = np.stack([x[1] for x in found_orgs], axis=0)
    all_used_conn = np.stack([x[2] for x in found_orgs], axis=0)
    all_tts = np.stack([x[3] for x in found_orgs], axis=0)
    all_out_nodes = np.array([x[4] for x in found_orgs])
    all_in_nodes = np.stack([x[5] for x in found_orgs], axis=0)
    all_comp_times = np.array([x[6] for x in found_orgs])
    return all_functions, all_conn, all_used_conn, all_tts, all_out_nodes, all_in_nodes, all_comp_times


def eval_natural_task(state, output_bits, truth_tables):
    output_bits = np.expand_dims(np.expand_dims(np.expand_dims(np.expand_dims(output_bits, -1), -1), 0), 0)
    truth_tables = truth_tables.T
    state_slice = np.squeeze(np.take_along_axis(state, output_bits, axis=-1), axis=-1)
    truth_tables = np.broadcast_to(np.expand_dims(np.expand_dims(truth_tables, -1), 0), state_slice.shape)
    return np.mean(np.bitwise_not(np.equal(truth_tables, state_slice)), axis=(0, 1))
