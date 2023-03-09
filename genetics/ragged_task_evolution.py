try:
    import cupy as cp
except:
    import numpy as cp

import numpy as np

import ragged_general_network
from genetics import graph_crossover


def sample_breeding_pairs_idx(population_batch_shape, n_children):
    return np.random.choice(population_batch_shape[-1],
                               (*population_batch_shape, n_children, 2))


def select_breeding_pairs_from_indicies(data, indicies):
    indicies_reshape_dim = (*np.shape(indicies)[:-1], 1, 1)
    first_parents = np.take_along_axis(data, np.reshape(indicies[..., 0], indicies_reshape_dim), -3)
    second_parents = np.take_along_axis(data, np.reshape(indicies[..., 1], indicies_reshape_dim), -3)
    return first_parents, second_parents


def sample_breeding_pairs(data, n_children):
    indicies = sample_breeding_pairs_idx(data.shape[:-3], n_children)
    return select_breeding_pairs_from_indicies(data, indicies)


def pair_breed_swap(first_parents, second_parents,  from_first_mask):
    from_first_ind = np.argwhere(np.tile(from_first_mask, first_parents.shape[-1]) == 1)
    children = np.copy(second_parents)
    slices = tuple(from_first_ind[:, i] for i in range(np.ndim(children)))
    children[slices] = first_parents[slices]
    return children


def pair_breed_swap_all(function_parents, connectivity_parents, used_connectivity_parents, p_first=0.5):
    batch_shape = function_parents[0].shape[:-1]
    from_first_mask = np.expand_dims(np.random.binomial(1, p_first, batch_shape), -1)
    return pair_breed_swap(*function_parents, from_first_mask),\
        pair_breed_swap(*connectivity_parents, from_first_mask),\
        pair_breed_swap(*used_connectivity_parents, from_first_mask)


def graph_crossover_breed(function_parents, connectivity_parents, used_connectivity_parents, special_nodes):
    N = function_parents[0].shape[-2]
    connectivity_parents_1, connectivity_parents_2 = connectivity_parents
    used_connectivity_parents_1, used_connectivity_parents_2 = used_connectivity_parents
    using_cuda = False
    if not isinstance(connectivity_parents[0], np.ndarray):
        using_cuda = True
        connectivity_parents_1 = cp.asnumpy(connectivity_parents_1)
        connectivity_parents_2 = cp.asnumpy(connectivity_parents_2)
        used_connectivity_parents_1 = cp.asnumpy(used_connectivity_parents_1)
        used_connectivity_parents_2 = cp.asnumpy(used_connectivity_parents_2)
    function_children = function_parents[0].copy()
    connectivity_children = connectivity_parents_1.copy()
    used_connectivity_children = used_connectivity_parents_1.copy()
    for batch_idx in range(connectivity_parents_1.shape[0]):
        for parent_idx in range(connectivity_parents_1.shape[1]):
            size_first = np.random.randint(1, N - len(special_nodes) - 1)

            this_connectivity, this_used_connectivity, org_0_map, org_1_map = graph_crossover.network_crossover_random(
                connectivity_parents_1[batch_idx, parent_idx, ...], used_connectivity_parents_1[batch_idx, parent_idx, ...],
                connectivity_parents_2[batch_idx, parent_idx, ...], used_connectivity_parents_2[batch_idx, parent_idx, ...],
                special_nodes, size_first)

            this_functions = graph_crossover.merge_functions(function_parents[0][batch_idx, parent_idx, ...],
                                                             function_parents[1][batch_idx, parent_idx, ...],
                                                             org_0_map, org_1_map)
            connectivity_children[batch_idx, parent_idx, ...] = this_connectivity
            used_connectivity_children[batch_idx, parent_idx, ...] = this_used_connectivity
            function_children[batch_idx, parent_idx, ...] = this_functions
    if using_cuda:
        connectivity_children = cp.array(connectivity_children)
        used_connectivity_children = cp.array(used_connectivity_children)
    return function_children, connectivity_children, used_connectivity_children


def pair_breed_random(first_parents, second_parents, p_first=0.5):
    from_first_ind = np.argwhere(np.random.binomial(1, p_first, first_parents.shape) == 1)
    children = np.copy(second_parents)
    slices = tuple(from_first_ind[:, i] for i in range(np.ndim(children)))
    children[slices] = first_parents[slices]
    return children


def split_breeding(first_parents, second_parents, n_children):
    mix_children = int(n_children/2)
    return np.concatenate([pair_breed_swap(first_parents[:, :mix_children, :, :], second_parents[:, :mix_children, :, :]),
                           pair_breed_random(first_parents[:, mix_children:, :, :], second_parents[:, mix_children:, :, :])], 1)


def split_breed_data(data, selected_parents, n_children):
    data_parents = select_breeding_pairs_from_indicies(data, selected_parents)
    children = split_breeding(*data_parents, n_children)
    return children


def mutate_binary(data, mutation_rate):
    xp = np
    if isinstance(data, cp.ndarray):
        xp = cp
    return np.bitwise_xor(data, xp.random.binomial(1, mutation_rate, data.shape).astype(data.dtype))


def mutate_integer(data, mutation_rate, rollover):
    xp = np
    if isinstance(data, cp.ndarray):
        xp = cp
    return np.mod(data + xp.random.binomial(1, mutation_rate, data.shape).astype(data.dtype), rollover)


def mutate_equal_prob(functions, connectivity, used_connectivity, mutation_rate):
    N = functions.shape[-2]
    return mutate_binary(functions, mutation_rate),\
        mutate_integer(connectivity, mutation_rate, N),\
        mutate_binary(used_connectivity, mutation_rate)


def dynamics_with_state_noise(input_state, functions, connections, used_connections, n_timesteps, p_noise):
    xp = np
    if isinstance(input_state, cp.ndarray):
        xp = cp
    current_state = input_state
    for _ in range(n_timesteps):
        current_state = np.bitwise_xor(ragged_general_network.ragged_k_state_update(current_state, functions, connections, used_connections),
                                       xp.random.binomial(1, p_noise, current_state.shape).astype(np.bool_))
    return ragged_general_network.ragged_k_state_update(current_state, functions, connections, used_connections)


def dynamics_with_function_noise(input_state, functions, connections, used_connections, n_timesteps, p_noise):
    xp = np
    if isinstance(input_state, cp.ndarray):
        xp = cp
    current_state = input_state
    for _ in range(n_timesteps):
        current_state = ragged_general_network.ragged_k_state_update(
            current_state, np.bitwise_xor(functions, xp.random.binomial(1, p_noise, functions.shape).astype(np.bool_)),
            connections, used_connections)
    return current_state


def dynamics_with_connectivity_noise(input_state, functions, connections, used_connections, n_timesteps, p_noise):
    xp = np
    if isinstance(input_state, cp.ndarray):
        xp = cp
    current_state = input_state
    for _ in range(n_timesteps):
        current_state = ragged_general_network.ragged_k_state_update(
            current_state, functions,
            connections, np.bitwise_xor(used_connections,  xp.random.binomial(1, p_noise, used_connections.shape).astype(np.bool_)))
    return current_state


def evaluate_populations(input_states, n_trajectories, functions, connectivity,
                         used_connectivity, n_timesteps, noise_prob, f_eval, dynamics_fn=dynamics_with_state_noise):
    return f_eval(
        dynamics_fn(np.broadcast_to(np.expand_dims(input_states, 0), (n_trajectories, *input_states.shape)),
                             functions, connectivity, used_connectivity, n_timesteps, noise_prob))


def evolutionary_step(input_states, n_trajectories, functions, connectivity, used_connectivity, n_timesteps, noise_prob, f_eval, breeding_fn, n_children, mutation_fn, dynamics_fn=dynamics_with_state_noise):
    n_populations = input_states.shape[-3]
    pop_size = input_states.shape[-2]
    keep_best = pop_size - n_children

    error_rates = evaluate_populations(input_states, n_trajectories, functions, connectivity, used_connectivity, n_timesteps, noise_prob, f_eval, dynamics_fn=dynamics_fn)

    sorted_error_rates = np.argsort(error_rates, axis=-1)
    expand_sort_error_rates = np.expand_dims(np.expand_dims(sorted_error_rates, -1), -1)

    parents = sample_breeding_pairs_idx([n_populations], n_children)

    best_functions = np.take_along_axis(functions, expand_sort_error_rates, 1)[:, :keep_best, :, :]
    best_connectivity = np.take_along_axis(connectivity, expand_sort_error_rates, 1)[:, :keep_best, :, :]
    best_used_connectivity = np.take_along_axis(used_connectivity, expand_sort_error_rates, 1)[:, :keep_best, :, :]

    function_children, connectivity_children, used_connectivity_children = breeding_fn(
        select_breeding_pairs_from_indicies(best_functions, parents),
        select_breeding_pairs_from_indicies(best_connectivity, parents),
        select_breeding_pairs_from_indicies(best_used_connectivity, parents))

    mutated_function_children, mutated_connectivity_children, mutated_used_connectivity_children = mutation_fn(
        function_children, connectivity_children, used_connectivity_children)

    functions = np.concatenate([best_functions, mutated_function_children], axis=1)
    connectivity = np.concatenate([best_connectivity, mutated_connectivity_children], axis=1)
    used_connectivity = np.concatenate([best_used_connectivity, mutated_used_connectivity_children], axis=1)

    population_errors = np.mean(error_rates, axis=1)
    return functions, connectivity, used_connectivity, population_errors




