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

