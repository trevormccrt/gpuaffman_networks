import numpy as np


import binary_core

_3_BIT_TT = binary_core.truth_table_columns(3)
_4_BIT_TT = binary_core.truth_table_columns(4)
_6_BIT_TT = binary_core.truth_table_columns(6)

_PAND_DESIRED_OUT = np.stack([np.bitwise_and(_4_BIT_TT[:, 0], _4_BIT_TT[:, 1]),
                              np.bitwise_and(_4_BIT_TT[:, 2], _4_BIT_TT[:, 3])])

_SHARED_AND_DESIRED_OUT = np.stack([np.bitwise_and(_3_BIT_TT[:, 0], _3_BIT_TT[:, 1]),
                                    np.bitwise_and(_3_BIT_TT[:, 1], _3_BIT_TT[:, 2])])

_SEQUENTIAL_AND_DESIERD_OUT = np.stack([np.bitwise_and(_3_BIT_TT[:, 0], _3_BIT_TT[:, 1]),
                                        np.bitwise_and(np.bitwise_and(_3_BIT_TT[:, 0], _3_BIT_TT[:, 1]), _3_BIT_TT[:, 2])])

_MODULAR_6_DESIRED_OUT = np.bitwise_or(np.bitwise_and(np.bitwise_or(_6_BIT_TT[:, 0], _6_BIT_TT[:, 1]),
                                                      np.bitwise_or(_6_BIT_TT[:, 2], _6_BIT_TT[:, 3])),
                                       np.bitwise_and(_6_BIT_TT[:, 4], _6_BIT_TT[:, 5]))


try:
    import cupy as cp
    _PAND_DESIRED_OUT_GPU = cp.array(_PAND_DESIRED_OUT)
    _SHARED_AND_DESIRED_OUT_GPU = cp.array(_SHARED_AND_DESIRED_OUT)
    _SEQUENTIAL_AND_DESIERD_OUT_GPU = cp.array(_SEQUENTIAL_AND_DESIERD_OUT)
    _MODULAR_6_DESIRED_OUT_GPU = cp.array(_MODULAR_6_DESIRED_OUT)

except:
    pass


def make_2_bit_input_state(N):
    input_state = np.zeros((4, N), dtype=np.uint8)
    input_state[0, (0, 1)] = (False, False)
    input_state[1, (0, 1)] = (False, True)
    input_state[2, (0, 1)] = (True, False)
    input_state[3, (0, 1)] = (True, True)
    return input_state


def evaluate_and_task(data):
    error_rate_0 = np.mean(np.equal(data[:, 0, :, :, 2], False), axis=0)
    error_rate_1 = np.mean(np.equal(data[:, 1, :, :, 2], False), axis=0)
    error_rate_2 = np.mean(np.equal(data[:, 2, :, :, 2], False), axis=0)
    error_rate_3 = np.mean(np.equal(data[:, 3, :, :, 2], True), axis=0)
    return np.mean(np.stack([error_rate_0, error_rate_1, error_rate_2, error_rate_3], axis=-1), axis=-1)


def evaluate_xor_task(data):
    error_rate_0 = np.mean(np.equal(data[:, 0, :, :, 2], True), axis=0)
    error_rate_1 = np.mean(np.equal(data[:, 1, :, :, 2], False), axis=0)
    error_rate_2 = np.mean(np.equal(data[:, 2, :, :, 2], False), axis=0)
    error_rate_3 = np.mean(np.equal(data[:, 3, :, :, 2], True), axis=0)
    return np.mean(np.stack([error_rate_0, error_rate_1, error_rate_2, error_rate_3], axis=-1), axis=-1)


def make_3_bit_input_state(N):
    input_state = np.zeros((8, N), dtype=np.uint8)
    input_state[:, :3] = _3_BIT_TT
    return input_state


def make_4_bit_input_state(N):
    input_state = np.zeros((16, N), dtype=np.uint8)
    input_state[:, :4] = _4_BIT_TT
    return input_state


def make_6_bit_input_state(N):
    input_state = np.zeros((64, N), dtype=np.uint8)
    input_state[:, :6] = _6_BIT_TT
    return input_state


def evaluate_pnand_task(data):
    data = np.moveaxis(data, 1, -2)
    desired_out = _PAND_DESIRED_OUT
    if not isinstance(data, np.ndarray):
        desired_out = _PAND_DESIRED_OUT_GPU
    error_rate = np.mean(np.equal(data[:, :, :, :, (4, 5)], desired_out.T), axis=(0, -2, -1))
    return error_rate


def evaluate_shared_and_task(data):
    data = np.moveaxis(data, 1, -2)
    desired_out = _SHARED_AND_DESIRED_OUT
    if not isinstance(data, np.ndarray):
        desired_out = _SHARED_AND_DESIRED_OUT_GPU
    error_rate = np.mean(np.equal(data[:, :, :, :, (3, 4)], desired_out.T), axis=(0, -2, -1))
    return error_rate


def evaluate_sequential_and_task(data):
    data = np.moveaxis(data, 1, -2)
    desired_out = _SEQUENTIAL_AND_DESIERD_OUT
    if not isinstance(data, np.ndarray):
        desired_out = _SEQUENTIAL_AND_DESIERD_OUT_GPU
    error_rate = np.mean(np.equal(data[:, :, :, :, (3, 4)], desired_out.T), axis=(0, -2, -1))
    return error_rate


def evaluate_modular_6_task(data):
    data = np.moveaxis(data, 1, -2)
    desired_out = _MODULAR_6_DESIRED_OUT
    if not isinstance(data, np.ndarray):
        desired_out = _MODULAR_6_DESIRED_OUT_GPU
    error_rate = np.mean(np.equal(data[:, :, :, :, 6], desired_out.T), axis=(0, -2))
    return error_rate


