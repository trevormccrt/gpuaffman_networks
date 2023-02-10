import numpy as np
import cupy as cp

import binary_core, cuda_binary_core


def test_binary_to_int():
    binary_dimension = 3
    data_b = binary_core.random_binary_data((20, 10, binary_dimension), 0.5)
    data_u = binary_core.binary_to_uint8(data_b, -1)
    data_u_c = cuda_binary_core.binary_to_uint8(cp.array(data_b))
    np.testing.assert_equal(data_u, cp.asnumpy(data_u_c))


def test_apply_binary_function():
    data = binary_core.random_binary_data((1000, 10, 3), 0.5)
    function = binary_core.random_binary_data((1000, 10, 8), 0.5)
    result_np = binary_core.apply_binary_function(data, function)
    result_cp = cuda_binary_core.apply_binary_function(cp.array(data), cp.array(function))
    np.testing.assert_equal(result_np, cp.asnumpy(result_cp))

