from rrt_base import RRT
from rrt_star import RRTStar
import numpy as np
import pytest


@pytest.fixture
def start():
    return np.array([11, 0])  # Start location


@pytest.fixture
def goal():
    return np.array([6, 8])  # Goal location


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
    return np.array([-2, 15])  # Bounds in both x and y


def test_run_rrt(start, goal, bounds, obstacles, plt):
    np.random.seed(7)
    rrt = RRT(start=start, goal=goal, bounds=bounds, obstacle_list=obstacles)
    rrt.plot()
    rrt.draw_graph()


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
    rrt_star.draw_graph()
