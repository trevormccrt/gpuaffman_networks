import numpy as np

from genetics import ragged_task_evolution


def test_sample_breeding_pairs():
    batch_dim = 10
    pop_dim = 100
    org_dim = 7
    data_dim = 4
    n_children = 3
    data = np.random.randint(0, 20, (batch_dim, pop_dim, org_dim, data_dim))
    first_parents, second_parents = ragged_task_evolution.sample_breeding_pairs(data, n_children)
    assert not np.all(np.equal(first_parents, second_parents))
    for parents in first_parents, second_parents:
        for child in range(n_children):
            assert np.all(np.any(np.all(np.all(np.expand_dims(parents[:, child, :, :], 1) == data, axis=-1), axis=-1), axis=-1))


def test_pair_breeding_function_swap():
    population_size = 100
    org_dim = 10
    batch_size = 3
    data_dim = 34
    n_children = 7
    data = np.random.randint(0, 20, (batch_size, population_size, org_dim, data_dim))
    first_parents, second_parents = ragged_task_evolution.sample_breeding_pairs(data, n_children)
    batch_shape = first_parents.shape[:-1]
    from_first_mask = np.expand_dims(np.random.binomial(1, 0.5, batch_shape), -1)
    children = ragged_task_evolution.pair_breed_swap(first_parents, second_parents, from_first_mask)
    assert not np.all(np.equal(children, first_parents))
    assert not np.all(np.equal(children, second_parents))
    for function_batch, children_batch in zip(np.concatenate([first_parents, second_parents], axis=1), children):
        for child in children_batch:
            for function in child:
                assert np.any(np.all(np.equal(function, function_batch), axis=-1))


def test_pair_breeding_random():
    population_size = 100
    org_dim = 10
    batch_size = 3
    data_dim = 34
    n_children = 7
    data = np.random.randint(0, 20, (batch_size, population_size, org_dim, data_dim))
    first_parents, second_parents = ragged_task_evolution.sample_breeding_pairs(data, n_children)
    children = ragged_task_evolution.pair_breed_random(first_parents, second_parents)
    assert not np.all(np.equal(children, first_parents))
    assert not np.all(np.equal(children, second_parents))
    for children_batch, first_parent_batch, second_parent_batch in zip(children, first_parents, second_parents):
        for child in children_batch:
            for function in child:
                assert np.any(np.all(np.bitwise_or(np.equal(function, first_parent_batch), np.equal(function, second_parent_batch)), axis=-1))
