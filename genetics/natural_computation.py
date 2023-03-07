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
    cycle_lengths = np.reshape(cycle_lengths, input_state.shape[:-1]).T
    output_nodes = []
    effective_truth_tables = []
    computation_stablilize_time = []
    non_empty_idx = []
    for i, (all_input_results, lengths) in enumerate(zip(found_cycles, cycle_lengths)):
        try:
            frozen_nodes = np.array([np.all(x == x[0], axis=0) for x in all_input_results])
            frozen_vales = np.array([x[0] for x in all_input_results])
            frozen_always = np.all(frozen_nodes, axis=0)
            doing_computation = np.bitwise_not(np.all(frozen_vales == frozen_vales[0], axis=0))
            comp_idx = np.argwhere(np.bitwise_and(doing_computation, frozen_always)).flatten()
            effective_truth_table = frozen_vales[:, comp_idx]
            output_nodes.append(comp_idx)
            effective_truth_tables.append(effective_truth_table)
            computation_stablilize_time.append(np.max(lengths))
            if len(comp_idx):
                non_empty_idx.append(i)
        except TypeError:
            output_nodes.append([])
            effective_truth_tables.append([])
            computation_stablilize_time.append(0)
    return output_nodes, effective_truth_tables, computation_stablilize_time, non_empty_idx


def order_natural_computations_by_rank(output_nodes, effective_truth_tables, computation_stabilize_time, computing_idx):
    computing_tts = [effective_truth_tables[x] for x in computing_idx]
    computing_output_nodes = [output_nodes[x] for x in computing_idx]
    computing_times = [computation_stabilize_time[x] for x in computing_idx]
    bits_used = [[analysis_util.compute_influence(y) > 0 for y in x.T] for x in computing_tts]
    ranks = [[np.sum(y) for y in x] for x in bits_used]
    rank_order = np.argsort([np.mean(x) for x in ranks])[::-1]
    ordered_ranks = [ranks[x] for x in rank_order]
    ordered_tts = [computing_tts[x] for x in rank_order]
    ordered_out_nodes = [computing_output_nodes[x] for x in rank_order]
    ordered_times = [computing_times[x] for x in rank_order]
    ordered_idx = [computing_idx[x] for x in rank_order]
    return ordered_ranks, ordered_tts, ordered_out_nodes, ordered_times, ordered_idx
