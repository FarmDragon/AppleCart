from time import sleep
import webbrowser
from manipulation.meshcat_cpp_utils import StartMeshcat, AddMeshcatTriad

from rrt.rrt_motion_planning import (
    IiwaProblem,
    ManipulationStationSim,
    rrt_planning,
    IKSolver,
)
from pydrake.all import RotationMatrix, RigidTransform
import numpy as np


def test_notebook():
    meshcat = StartMeshcat()
    env = ManipulationStationSim(True, meshcat=meshcat)
    q_start = env.q0
    R_WG = RotationMatrix(np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]]).T)
    T_WG_goal = RigidTransform(
        p=np.array([4.69565839e-01, 2.95894043e-16, 0.65]), R=R_WG
    )
    AddMeshcatTriad(meshcat, "goal pose", X_PT=T_WG_goal, opacity=0.5)

    ik_solver = IKSolver()
    q_goal, optimal = ik_solver.solve(T_WG_goal, q_guess=q_start)

    gripper_setpoint = 0.1
    door_angle = np.pi / 2 - 0.001
    left_door_angle = -np.pi / 2
    right_door_angle = np.pi / 2

    iiwa_problem = IiwaProblem(
        q_start=q_start,
        q_goal=q_goal,
        gripper_setpoint=gripper_setpoint,
        left_door_angle=left_door_angle,
        right_door_angle=right_door_angle,
        is_visualizing=True,
        meshcat=meshcat,
    )
    webbrowser.open(meshcat.web_url())
    path = rrt_planning(iiwa_problem, 600, 0.05)
    sleep(8)
    iiwa_problem.visualize_path(path)
