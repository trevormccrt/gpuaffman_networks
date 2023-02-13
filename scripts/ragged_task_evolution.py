import numpy as np

import ragged_general_network


def run_dynamics_forward(input_state, functions, connections, used_connections, n_timesteps, p_noise):
    current_state = input_state
    for _ in range(n_timesteps):
        current_state = np.bitwise_xor(ragged_general_network.ragged_k_state_update(current_state, functions, connections, used_connections),
                                       np.random.binomial(1, p_noise, current_state.shape).astype(np.bool_))
    return ragged_general_network.ragged_k_state_update(current_state, functions, connections, used_connections)


def evolutionary_step(input_states, functions, connectivity, used_connectivity, n_timesteps, noise_prob, f_eval, breeding_fn, binary_mutation_fn, integer_mutation_fn):
    updated_states = run_dynamics_forward(input_states, functions, connectivity, used_connectivity,
                                          n_memory_timesteps + np.random.randint(0, 3), noise_prob)
    error_rates = f_eval(updated_states)
    sorted_error_rates = cp.argsort(error_rates, axis=-1)
    expand_sort_error_rates = cp.expand_dims(cp.expand_dims(sorted_error_rates, -1), -1)

    parents = genetics_util.sample_breeding_pairs_idx([n_populations], n_children)

    best_functions = cp.take_along_axis(functions, expand_sort_error_rates, 1)[:, :keep_best, :, :]
    best_connectivity = cp.take_along_axis(connectivity, expand_sort_error_rates, 1)[:, :keep_best, :, :]
    best_used_connectivity = cp.take_along_axis(used_connectivity, expand_sort_error_rates, 1)[:, :keep_best, :, :]

    function_children = mutate_binary(breed_data(best_functions, parents, n_children), mutation_rate)
    connectivity_children = mutate_integer(breed_data(best_connectivity,
                                                      parents, n_children), mutation_rate, N)
    used_connectivity_children = mutate_binary(breed_data(used_connectivity,
                                                          parents, n_children), mutation_rate)

    functions = cp.concatenate([best_functions, function_children], axis=1)
    connectivity = cp.concatenate([best_connectivity, connectivity_children], axis=1)
    used_connectivity = cp.concatenate([best_used_connectivity, used_connectivity_children], axis=1)

    population_errors = cp.mean(error_rates, axis=1)


for generation in range(n_generations):

    if generation % 100 == 0:
        print("GENERATION {}".format(generation))
    for i, error in enumerate(population_errors):
        if error < best_errors[i]:
            best_errors[i] = error
            best_populations[i] = (cp.asnumpy(best_functions[i]), cp.asnumpy(best_connectivity[i]), cp.asnumpy(best_used_connectivity[i]))
            print("population {} mean error rate: {}".format(i, error))

