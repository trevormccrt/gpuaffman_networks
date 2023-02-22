import numpy as np

import analysis_util


def test_influence():
    test_f_1 = np.array([False, True, True, True, False, False, False, True], dtype=np.bool_)
    inf = analysis_util.compute_influence(test_f_1)
    np.testing.assert_equal(inf, [0.5, 0.5, 0.5])
    test_f_2 = np.array([False, True, False, True])
    inf = analysis_util.compute_influence(test_f_2)
    np.testing.assert_equal(inf, [1, 0])

