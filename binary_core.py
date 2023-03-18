import numpy as np

_BYTE_CONV = np.array([1 << x for x in range(8)], dtype=np.uint8)


def binary_to_uint8(data):
    """Convert a binary tensor to a uint8 tensor along it's last axis.

    Args:
        data: a binary tensor to be converted to unit8.
    Returns:
        A tensor of the same shape as `data` with the last axis removed holding the equivalent uint8 values.
    """
    return np.einsum("...k, k -> ...", data, _BYTE_CONV[:data.shape[-1]], dtype=np.uint8)


def truth_table_inputs(dimension):
    """Compute the input labels to a truth table of length 2^`dimension`.

    Args:
        dimension: the truth table dimension in bits.
    Returns:
        A tensor of shape [2^`dimension`, dimension] containing the input bitstrings for a truth table.
    """
    return np.unpackbits(np.expand_dims(np.arange(start=0, stop=1<<dimension, step=1).astype(np.uint8), -1),
                         axis=-1, bitorder="little").astype(np.bool_)[..., :dimension]


def apply_binary_function(binary_data, functions):
    """Apply an arbitrary binary function to some data.

    Apply an arbitrary binary function to the last axis of `binary_data`. The function(s) to apply are specified in the
    form of truth tables by `functions`, the batch shape of which must be broadcastable to the shape of `binary_data`.

    Args:
        binary_data: a binary tensor with shape [..., d].
        functions: a binary tensor with final dimension lengths 2^d. The other dimensions have to be broadcastable to
            the shape of `binary_data`.
    Returns:
        A tensor of shape [...], containing the result of applying `functions` to `binary_data`.
    """
    return np.squeeze(np.take_along_axis(np.broadcast_to(functions, (*binary_data.shape[:-1], functions.shape[-1])),
                                         np.expand_dims(binary_to_uint8(binary_data), -1), -1), -1)
