import cupy as cp

import genetics_util


def test_sample_breeding_pairs():
    batch_dim = 10
    pop_dim = 100
    org_dim = 7
    data_dim = 4
    n_children = 3
    data = cp.random.randint(0, 20, (batch_dim, pop_dim, org_dim, data_dim))
    first_parents, second_parents = ragged_genetics.sample_breeding_pairs(data, n_children)
    for parents in first_parents, second_parents:
        for child in range(n_children):
            assert cp.all(cp.any(cp.all(cp.all(cp.expand_dims(parents[:, child, :, :], 1) == data, axis=-1), axis=-1), axis=-1))

def test_pair_breeding_function_swap():
    population_size = 15
    automaton_size = 10
    batch_size = 3
    bit_length = 34
    for dimension in (1, 2, 3):
        functions = binary_core.random_binary_data((batch_size, population_size, *tuple([automaton_size] * dimension), bit_length), 0.5)
        children = binary_cellular_automaton.pair_breed_mix_functions(functions, 7, automaton_dimension=dimension)
        for function_batch, children_batch in zip(functions, children):
            flat_children = np.reshape(children_batch, (-1, np.shape(children_batch)[-1]))
            for child in flat_children:
                assert np.any(np.all(child == function_batch, axis=-1))


def test_pair_breeding_random():
    population_size = 15
    automaton_size = 10
    batch_size = 3
    bit_length = 34
    for dimension in (1, 2, 3):
        functions = binary_core.random_binary_data(
            (batch_size, population_size, *tuple([automaton_size] * dimension), bit_length), 0.5)
        _ = binary_cellular_automaton.pair_breed_random(functions, 7, automaton_dimension=dimension)
