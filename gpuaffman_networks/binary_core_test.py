import cupy as cp
import numpy as np

from gpuaffman_networks import binary_core


def _uint8_to_binary(data, n_bits=8):
    return np.unpackbits(np.expand_dims(data, -1), axis=-1, bitorder="little")[..., :n_bits].astype(np.bool_)


def test_conversion():
    binary_dimension = 3
    data_b = np.random.binomial(1, 0.5, (20, 10, binary_dimension)).astype(np.bool_)
    data_u = binary_core.binary_to_uint8(data_b)
    data_b_re = _uint8_to_binary(data_u, binary_dimension)
    np.testing.assert_equal(data_b_re, data_b)


def test_function_apply():
    data_b = np.random.binomial(1, 0.5, (20, 10, 2)).astype(np.bool_)
    function_and = np.array([False, False, False, True])
    function_or = np.array([False, True, True, True])
    test_and = binary_core.apply_binary_function(data_b, function_and)
    test_or = binary_core.apply_binary_function(data_b, function_or)
    true_and = np.logical_and(data_b[..., 0], data_b[..., 1])
    true_or = np.logical_or(data_b[..., 0], data_b[..., 1])
    np.testing.assert_equal(test_and, true_and)
    np.testing.assert_equal(test_or, true_or)
    data_u = binary_core.binary_to_uint8(data_b)
    twos = np.where(data_u==2, True, False)
    fn_three = np.array([False, False, True, False])
    twos_test = binary_core.apply_binary_function(data_b, fn_three)
    np.testing.assert_equal(twos, twos_test)


def test_vectorized_function_apply():
    for i in range(10):
        data_b = np.random.binomial(1, 0.5, (2, 2)).astype(np.bool_)
        function_and = np.array([False, False, False, True])
        function_or = np.array([False, True, True, True])
        vectorized_functions = np.stack([function_and, function_or], axis=0)
        applied = binary_core.apply_binary_function(data_b, vectorized_functions)
        true = [np.logical_and(data_b[0, 0], data_b[0, 1]), np.logical_or(data_b[1, 0], data_b[1, 1])]
        np.testing.assert_equal(applied, true)


def test_truth_table_columns():
    binary_dimension = 3
    tt = binary_core.truth_table_inputs(binary_dimension)
    test_binary = np.random.binomial(1, 0.5, (20, 10, binary_dimension)).astype(np.bool_)
    test_u = binary_core.binary_to_uint8(test_binary)
    tt_samples = np.take(tt, test_u, axis=0)
    np.testing.assert_equal(test_binary, tt_samples)


def test_cuda():
    batch_size = 100
    function_size = 4
    data = np.random.binomial(1, 0.5, size=(batch_size, function_size)).astype(np.bool_)
    functions = np.random.binomial(1, 0.5, size=(batch_size, 1<<function_size)).astype(np.bool_)

    data_cp = cp.array(data)
    functions_cp = cp.array(functions)

    result_np = binary_core.apply_binary_function(data, functions)
    result_cp = binary_core.apply_binary_function(data_cp, functions_cp)

    np.testing.assert_equal(result_np, cp.asnumpy(result_cp))



