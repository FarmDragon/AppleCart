import numpy as np
from value_iteration import *


def test_compute_cost():
    Q = np.diag([10, 1])
    data = np.array(
        [
            [2, 0, 0, 2],
            [0, 2, 0, 2],
        ]
    )
    target_state = np.array(
        [
            [2],
            [2],
        ]
    )

    # linear einsum
    np.testing.assert_equal(
        np.einsum("nm,mj->nj", Q, data),
        np.array(
            [
                [20, 0, 0, 20],
                [0, 2, 0, 2],
            ]
        ),
    )

    # quadratic einsum
    np.testing.assert_equal(
        np.einsum("nj,nm,mj->j", data, Q, data), np.array([40, 4, 0, 44])
    )

    # quadratic cost function
    np.testing.assert_equal(compute_quadratic_cost(Q, data), np.array([40, 4, 0, 44]))

    # difference
    np.testing.assert_equal(
        data - target_state,
        np.array(
            [
                [0, -2, -2, 0],
                [-2, 0, -2, 0],
            ]
        ),
    )

    # quadratic difference cost function
    np.testing.assert_equal(
        compute_state_cost(Q, target_state, data), np.array([4, 40, 44, 0])
    )
