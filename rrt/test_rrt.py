from audioop import reverse
from pathlib import Path
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


def test_run_rrt_from_notebook(plt):
    np.random.seed(7)
    rrt = RRT(
        start=np.array([11, 0]),
        goal=np.array([6, 8]),
        bounds=np.array([-2, 15]),
        obstacle_list=[  # circles parametrized by [x, y, radius]
            np.array([9, 6, 2]),
            np.array([9, 8, 1]),
            np.array([9, 10, 2]),
            np.array([4, 5, 2]),
            np.array([7, 5, 2]),
            np.array([4, 10, 1]),
        ],
        max_extend_length=3.0,
        path_resolution=0.5,
        goal_sample_rate=0.05,
        max_iter=100,
        plt=plt,
    )
    rrt.plot()


def test_run_rrt_star_from_notebook(plt):
    np.random.seed(7)
    rrt = RRTStar(
        start=np.array([11, 0]),
        goal=np.array([6, 8]),
        bounds=np.array([-2, 15]),
        obstacle_list=[  # circles parametrized by [x, y, radius]
            np.array([9, 6, 2]),
            np.array([9, 8, 1]),
            np.array([9, 10, 2]),
            np.array([4, 5, 2]),
            np.array([7, 5, 2]),
            np.array([4, 10, 1]),
        ],
        max_extend_length=5.0,
        path_resolution=0.5,
        goal_sample_rate=0.0,
        max_iter=5000,
        connect_circle_dist=50.0,
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
    def __init__(self, dt=0.01, m=1, g=9.8, l=0.5, b=0.1, u_max=2) -> None:
        self.dt = dt
        self.u_max = u_max
        self.m = m
        self.g = g
        self.l = l
        self.b = b

    def calculate_u(self, f, t):
        # if t[1] > f[1]:
        #     return self.u_max
        # else:
        #     return -self.u_max
        return (
            self.m * self.l**2 * (t[1] - f[1]) / self.dt
            + np.sin(f[0]) * self.m * self.g * self.l
            + self.b * f[0]
        )

    def run_forward(self, f, t):
        u = self.calculate_u(f, t)
        u_star = bound(-self.u_max, self.u_max, u)
        new_node = RRT.Node(self.x_new(f, u))
        new_node.u = u_star
        return new_node

    def x_new(self, x, u) -> np.array:
        return np.array(
            [
                x[0] + x[1] * self.dt,
                (
                    (u - self.b * x[1]) / (self.m * self.l**2)
                    - self.g * np.sin(x[0]) / self.l
                )
                * self.dt
                + x[1],
            ]
        )

    def cost(self, f, t):
        return self.calculate_u(f, t)

    def calculate_reachable_states(self, x: np.array) -> list[np.array]:
        return set([self.x_new(x, self.u_max), self.x_new(x, self.u_max)])


@pytest.mark.parametrize(
    "x,u",
    [([np.pi - 0.05, 0], 0), ([0, 0], 3), ([0, 0], 4)],
    ids=["fall", "constant_limited_torque", "constant_escape_torque"],
)
def test_pendulum_fall(plt, x, u):
    start = x
    plt.figure()
    ax = plt.gca()
    plt.axis(
        [
            -10,
            10,
            -10,
            10,
        ]
    )
    plt.plot(start[0], start[1], "xr", markersize=10)
    plt.legend(("start"), loc="upper left")
    plt.gca().set_aspect("equal")
    plt.tight_layout()
    system = Pendulum(dt=0.01)
    current = start
    for i in range(1000):
        new = system.x_new(current, u)
        plt.plot([current[0], new[0]], [current[1], new[1]], "-g")
        current = new


def test_rrt_for_simple_pendulum_balance(bounds, plt):
    """Models a simple pendulum"""
    rrt = RRT(
        start=np.array([0, 0]),
        goal=np.array([np.pi, 0]),
        obstacle_list=[],
        bounds=np.array([-6, 6]),
        max_extend_length=0.1,
        max_iter=3500,
        dynamics=Pendulum(),
        plt=plt,
    )
    rrt.plan()
    rrt.plot()
    rrt.save()


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
    rrt_star.plan()
    rrt_star.plot()


def test_rrt_star_simple_pendulum(start, goal, bounds, plt):
    """Models a simple pendulum"""
    rrt_star = RRTStar(
        start=start,
        goal=goal,
        bounds=bounds,
        max_extend_length=0.2,
        max_iter=1000,
        dynamics=Pendulum(),
        connect_circle_dist=3,
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


def test_load_file():
    inputs_path = Path.cwd() / "rrt_path" / "inputs.npy"
    state_path = Path.cwd() / "rrt_path" / "states.npy"
    inputs = np.load(inputs_path, allow_pickle=True).tolist()
    states = np.load(state_path, allow_pickle=True).tolist()
    import json

    states.reverse()
    inputs.reverse()
    json.dump(
        {"states": states, "inputs": inputs},
        (Path.cwd() / "rrt_path" / "pendulum_swingup.json").open("w"),
    )
