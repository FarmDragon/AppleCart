from rrt.rrt_base import RRT
import numpy as np


def test_calculate_nearest_state():
    nearest_index = RRT().nearest_neighbor_index(
        np.array([0, 0]), np.array([[0, 1], [1, 1]])
    )
    assert nearest_index == 0
    nearest_index = RRT().nearest_neighbor_index(
        np.array([1, 1.5]), np.array([[0, 1], [1, 1]])
    )
    assert nearest_index == 1
