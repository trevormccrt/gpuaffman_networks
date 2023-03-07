import numpy as np


from genetics import natural_computation, analysis_util


def test_nat_comp():
    N = 40
    k_max = 3
    avg_k = 2.0
    batch_size = 1000
    functions = np.random.binomial(1, 0.5, (batch_size, N, 1 << k_max)).astype(np.bool_)
    connectivity = np.random.randint(0, N, (batch_size, N, k_max)).astype(np.uint8)
    used_connectivity = np.random.binomial(1, avg_k/k_max, (batch_size, N, k_max)).astype(np.bool_)
    output_nodes, effective_truth_tables, computation_stablilize_time, computing_idx =  natural_computation.natural_computation_search(functions, connectivity, used_connectivity, np.arange(start=0, stop=6, step=1))
    ordered_ranks, ordered_tts, ordered_out_nodes, ordered_times, ordered_idx = natural_computation.order_natural_computations_by_rank(output_nodes, effective_truth_tables, computation_stablilize_time, computing_idx)
    responsible_functions = functions[ordered_idx]
    responsible_connectivity = connectivity[ordered_idx]
    responsible_used_connectivity = used_connectivity[ordered_idx]

    print("")

test_nat_comp()
