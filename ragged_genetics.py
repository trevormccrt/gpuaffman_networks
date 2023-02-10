import cupy as cp
import numpy as np


def sample_breeding_pairs_idx(population_batch_shape, n_children):
    return cp.random.choice(population_batch_shape[-1],
                               (*population_batch_shape, n_children, 2))


def select_breeding_pairs_from_indicies(data, indicies):
    indicies_reshape_dim = (*cp.shape(indicies)[:-1], 1, 1)
    first_parents = cp.take_along_axis(data, cp.reshape(indicies[..., 0], indicies_reshape_dim), -3)
    second_parents = cp.take_along_axis(data, cp.reshape(indicies[..., 1], indicies_reshape_dim), -3)
    return first_parents, second_parents


def sample_breeding_pairs(data, n_children):
    indicies = sample_breeding_pairs_idx(data.shape[:-3], n_children)
    return select_breeding_pairs_from_indicies(data, indicies)


def pair_breed_mix(population, n_children, p_first=0.5):
    first_parents, second_parents = sample_breeding_pairs(population, n_children)
    from_first_ind = cp.argwhere(cp.tile(cp.expand_dims(cp.random.binomial(1, p_first, first_parents.shape[:-1]), -1), first_parents.shape[-1])==1)
    children = cp.copy(second_parents)
    slices = tuple(from_first_ind[:, i] for i in range(cp.ndim(children)))
    children[slices] = first_parents[slices]
    return children


def pair_breed_random(population, n_children, p_first=0.5, automaton_dimension=1):
    first_parents, second_parents = sample_breeding_pairs(population, n_children, automaton_dimension)
    from_first_ind = cp.argwhere(cp.random.binomial(1, p_first, first_parents.shape) == 1)
    children = cp.copy(second_parents)
    slices = tuple(from_first_ind[:, i] for i in range(cp.ndim(children)))
    children[slices] = first_parents[slices]
    return children
