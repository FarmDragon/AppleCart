import numpy as np
from value_iteration import *
from pydrake.examples.pendulum import PendulumPlant
from pydrake.all import Simulator


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


def setup_pendulum_state():
    plant = PendulumPlant()
    num_states = plant.num_continuous_states()
    num_inputs = 1
    num_samples = 50
    R_diag = np.array([2])

    theta_states = np.linspace(0, 2 * np.pi, num_samples)
    theta_dot_states = np.linspace(-10, 10, num_samples)

    state_grid = np.meshgrid(theta_states, theta_dot_states, indexing="ij")
    state_data = np.vstack([s.flatten() for s in state_grid])

    target_state = np.array([np.pi, 0.0]).reshape(-1, 1)
    state_data = np.hstack([state_data, target_state])

    num_state_data = state_data.shape[1]

    return (R_diag, num_states, num_inputs, num_state_data)


def test_compute_u_star():
    (R_diag, num_states, num_inputs, num_state_data) = setup_pendulum_state()

    dJdX = np.asfortranarray(np.random.randn(num_states, num_state_data))
    dstate_dynamics_du = np.random.randn(num_states, num_inputs, num_state_data)
    compute_u_star(R_diag, dJdX, dstate_dynamics_du)

    num_states = 2
    num_samples = 4
    num_inputs = 2

    R_diag = np.array([10, 1])
    dJdX = np.array(
        [
            [0, 1, 0, 1],
            [0, 0, 1, 1],
        ]
    )
    dstate_dynamics_du = np.array(
        [
            [[1, 2, 3, 4], [0, 0, 0, 0]],
            [[1, 2, 3, 4], [1, 2, 3, 4]],
        ]
    )

    np.testing.assert_equal(R_diag.shape, (num_inputs,))
    np.testing.assert_equal(dJdX.shape, (num_states, num_samples))

    # invert R, a diagonal matrix
    R_inverse = np.diag(1 / R_diag)
    np.testing.assert_equal(
        R_inverse,
        np.array(
            [
                [1 / 10, 0],
                [0, 1],
            ]
        ),
    )

    # multiply R inverse by a single dstate_dynamics_du
    f2 = dstate_dynamics_du[:, :, 1]
    np.testing.assert_equal(R_inverse.shape, (num_inputs, num_inputs))
    np.testing.assert_equal(f2.shape, (num_inputs, num_states))
    np.testing.assert_equal(np.einsum("uu,xu->ux", R_inverse, f2), R_inverse @ f2.T)

    # multiply R inverse by all dstate_dynamics_du (for all samples)
    np.testing.assert_equal(
        dstate_dynamics_du.shape, (num_states, num_inputs, num_samples)
    )
    answer = np.einsum("uu,xut->uxt", R_inverse, dstate_dynamics_du)
    # todo: what would I expect this to look like? create an assert
    # sort of like this, but not quite?
    # np.array([
    #         [[.1, .2, .3, .4], [1, 2, 3, 4]],
    #         [[0, 0, 0, 0], [1, 2, 3, 4]],
    # ])

    # multiply that by dJdX
    answer = np.einsum("uu,xut,xt->ut", R_inverse, dstate_dynamics_du, dJdX)
    np.testing.assert_equal(answer.shape, (num_inputs, num_samples))
    print(answer)
    # todo: test the expected output array
    # np.testing.assert_equal(
    #     answer,
    #     np.array([
    #         [0, 1/10, 0, 1/10],
    #         [0, 0, 1, 1],
    #     ])
    # )
