import numpy as np

from gpuaffman_networks import linear_network


def test_update_1d_local_periodic():
    rep_code = np.array([False, False, False, True, False, True, True, True])
    state = np.ones((20, 10, 5)).astype(bool)
    update = linear_network.state_update(state, rep_code)
    np.testing.assert_allclose(state, update)
