import cupy as cp


_BYTE_CONV = cp.array([1 << x for x in range(8)], dtype=cp.uint8)


def binary_to_uint8(data):
    return cp.einsum("...k, k -> ...", data, _BYTE_CONV[:data.shape[-1]], dtype=cp.uint8)


def apply_binary_function(binary_data, functions):
    return cp.squeeze(cp.take_along_axis(cp.broadcast_to(functions, (*binary_data.shape[:-1], functions.shape[-1])),
                                         cp.expand_dims(binary_to_uint8(binary_data), -1), -1), -1)


