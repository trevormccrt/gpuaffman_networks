import cupy as cp

import genetics_util, binary_core


def test_sample_breeding_pairs():
    batch_dim = 10
    pop_dim = 100
    org_dim = 7
    data_dim = 4
    n_children = 3
    data = cp.random.randint(0, 20, (batch_dim, pop_dim, org_dim, data_dim))
    first_parents, second_parents = genetics_util.sample_breeding_pairs(data, n_children)
    assert not cp.all(cp.equal(first_parents, second_parents))
    for parents in first_parents, second_parents:
        for child in range(n_children):
            assert cp.all(cp.any(cp.all(cp.all(cp.expand_dims(parents[:, child, :, :], 1) == data, axis=-1), axis=-1), axis=-1))


def test_pair_breeding_function_swap():
    population_size = 100
    org_dim = 10
    batch_size = 3
    data_dim = 34
    n_children = 7
    data = cp.random.randint(0, 20, (batch_size, population_size, org_dim, data_dim))
    first_parents, second_parents = genetics_util.sample_breeding_pairs(data, n_children)
    children = genetics_util.pair_breed_swap(first_parents, second_parents)
    assert not cp.all(cp.equal(children, first_parents))
    assert not cp.all(cp.equal(children, second_parents))
    for function_batch, children_batch in zip(cp.concatenate([first_parents, second_parents], axis=1), children):
        for child in children_batch:
            for function in child:
                a = cp.all(cp.equal(function, function_batch), axis=-1)
                assert cp.any(a)


def test_pair_breeding_random():
    population_size = 100
    org_dim = 10
    batch_size = 3
    data_dim = 34
    n_children = 7
    data = cp.random.randint(0, 20, (batch_size, population_size, org_dim, data_dim))
    first_parents, second_parents = genetics_util.sample_breeding_pairs(data, n_children)
    children = genetics_util.pair_breed_random(first_parents, second_parents)
    assert not cp.all(cp.equal(children, first_parents))
    assert not cp.all(cp.equal(children, second_parents))
    for children_batch, first_parent_batch, second_parent_batch in zip(children, first_parents, second_parents):
        for child in children_batch:
            for function in child:
                a = cp.all(cp.bitwise_or(cp.equal(function, first_parent_batch), cp.equal(function, second_parent_batch)), axis=-1)
                assert cp.any(a)

