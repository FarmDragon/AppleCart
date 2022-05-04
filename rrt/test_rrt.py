from rrt_base import RRT
from rrt_star import RRTStar
import numpy as np
import pytest


@pytest.fixture
def start():
    return np.array([0, 0])  # Start location


@pytest.fixture
def goal():
    return np.array([2, 2])  # Goal location


@pytest.fixture
def obstacles():
    return [  # circles parametrized by [x, y, radius]
        np.array([9, 6, 2]),
        np.array([9, 8, 1]),
        np.array([9, 10, 2]),
        np.array([4, 5, 2]),
        np.array([7, 5, 2]),
        np.array([4, 10, 1]),
    ]


@pytest.fixture
def bounds():
    return np.array([-4, 4])  # Bounds in both x and y


def test_run_rrt(start, goal, bounds, obstacles, plt):
    np.random.seed(7)
    rrt = RRT(
        start=start,
        goal=goal,
        bounds=bounds,
        obstacle_list=obstacles,
        plt=plt,
    )
    rrt.plot()


def test_rrt_star(start, goal, bounds, obstacles, plt):
    np.random.seed(7)
    rrt_star = RRTStar(
        start=start,
        goal=goal,
        bounds=bounds,
        obstacle_list=obstacles,
        max_iter=300,
        plt=plt,
    )
    print("Minimum cost: {}".format(rrt_star.min_cost))
    assert rrt_star.min_cost < 100
    rrt_star.plot()


def bound(low, high, value):
    return max(low, min(high, value))


class Pendulum(RRT.Dynamics):
    def __init__(self) -> None:
        self.dt = 0.1

    def calculate_u(self, f, t):
        u = (t[1] - f[1]) / self.dt + np.sin(f[0])
        return u

    def run_forward(self, f, t):
        u = (t[1] - f[1]) / self.dt + np.sin(f[0])
        u_star = bound(-0.3, 0.3, u)
        new_node = RRT.Node(
            np.array(
                [
                    f[0] + f[1] * self.dt,
                    f[1] - np.sin(f[0]) * self.dt + u_star * self.dt,
                ]
            )
        )
        new_node.u = u_star
        return new_node

    def cost(self, f, t):
        return self.calculate_u(f, t)


def test_rrt_for_simple_pendulum(start, goal, bounds, plt):
    """Models a simple pendulum"""
    rrt = RRT(
        start=start,
        goal=goal,
        bounds=bounds,
        max_extend_length=0.1,
        max_iter=3000,
        dynamics=Pendulum(),
        plt=plt,
    )
    rrt.plot()


@pytest.mark.skip(reason="no way of currently testing this")
def test_rg_rrt_for_simple_pendulum(start, goal, bounds, obstacles, plt):
    """Models a simple pendulum"""
    rrt_star = RRT(
        start=start,
        goal=goal,
        bounds=bounds,
        obstacle_list=obstacles,
        max_extend_length=0.2,
        max_iter=3000,
        dynamics=Pendulum(),
        plt=plt,
    )
    rrt_star.plot()


@pytest.mark.skip(reason="no way of currently testing this")
def test_rrt_star_simple_pendulum(start, goal, bounds, plt):
    """Models a simple pendulum"""
    rrt_star = RRTStar(
        start=start,
        goal=goal,
        bounds=bounds,
        max_extend_length=0.2,
        max_iter=10,
        dynamics=Pendulum(),
        plt=plt,
    )
    rrt_star.plot()


@pytest.mark.skip(reason="no way of currently testing this")
def test_rrt_star_custom_distance_function(start, goal, bounds, obstacles, plt):
    """This should model a car driving through a course - holonomic constraint."""
    # Add steering angle and velocity (both 0)
    start = [*start, 0, 0]
    goal = [*goal, 0, 0]

    # inputs are steering wheel acceleration and acceleration forward or backward.
    # The linearization of the system is simple.

    rrt_star = RRTStar(
        start=start,
        goal=goal,
        bounds=bounds,
        obstacle_list=obstacles,
        max_iter=10,
        distance_function=lambda a, b: np.linalg.norm(a - b),
        plt=plt,
    )
    print("Minimum cost: {}".format(rrt_star.min_cost))
    assert rrt_star.min_cost < 100
    rrt_star.plot()
