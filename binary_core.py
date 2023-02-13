import numpy as np

_BYTE_CONV = np.array([1 << x for x in range(8)], dtype=np.uint8)


def random_unit8(shape, n_bits=8):
    return np.random.randint(0, 1 << n_bits, shape).astype(np.uint8)


def random_binary_data(shape, p):
    return np.random.binomial(1, p, shape).astype(np.bool_)


def binary_to_uint8(data):
    return np.einsum("...k, k -> ...", data, _BYTE_CONV[:data.shape[-1]], dtype=np.uint8)


def uint8_to_binary(data, n_bits=8):
    return np.unpackbits(np.expand_dims(data, -1), axis=-1, bitorder="little")[..., :n_bits].astype(np.bool_)


def random_binary_function(dimension, p):
    return np.random.binomial(1, p, 1 << dimension).astype(np.bool_)


def truth_table_columns(dimension):
    return np.unpackbits(np.expand_dims(np.arange(start=0, stop=1<<dimension, step=1).astype(np.uint8), -1),
                         axis=-1, bitorder="little").astype(np.bool_)[..., :dimension]


def apply_binary_function(binary_data, functions):
    return np.squeeze(np.take_along_axis(np.broadcast_to(functions, (*binary_data.shape[:-1], functions.shape[-1])),
                                         np.expand_dims(binary_to_uint8(binary_data), -1), -1), -1)
