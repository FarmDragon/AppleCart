# %% [markdown]
# # Triple cart-pole

# %%
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pydot
from IPython.display import HTML, SVG, clear_output, display
from pydrake.all import (
    Box,
    DiagramBuilder,
    DirectCollocation,
    DirectTranscription,
    FiniteHorizonLinearQuadraticRegulatorOptions,
    GraphOfConvexSets,
    HPolyhedron,
    LinearSystem,
    LogVectorOutput,
    MakeFiniteHorizonLinearQuadraticRegulator,
    MathematicalProgram,
    MosekSolver,
    MultibodyPlant,
    MultibodyPositionToGeometryPose,
    Parser,
    PiecewisePolynomial,
    PlanarSceneGraphVisualizer,
    Point,
    PointCloud,
    Rgba,
    RigidTransform,
    RotationMatrix,
    SceneGraph,
    Simulator,
    Solve,
    Sphere,
    StartMeshcat,
    TrajectorySource,
    Variable,
    eq,
)
from pydrake.examples.acrobot import AcrobotGeometry, AcrobotPlant
from pydrake.examples.pendulum import PendulumPlant, PendulumState

from underactuated import FindResource, running_as_notebook
from underactuated.jupyter import AdvanceToAndVisualize
from underactuated.meshcat_utils import draw_points, set_planar_viewpoint
from underactuated.pendulum import PendulumVisualizer


# %%
NUM_BREAKPOINTS = 21

plant = MultibodyPlant(time_step=0.0)
scene_graph = SceneGraph()
plant.RegisterAsSourceForSceneGraph(scene_graph)
# file_name = FindResource("models/cartpole.urdf")
Parser(plant).AddModelFromFile("triple_cartpole.urdf")
plant.Finalize()

context = plant.CreateDefaultContext()
dircol = DirectCollocation(
    plant,
    context,
    num_time_samples=NUM_BREAKPOINTS,
    minimum_timestep=0.1,
    maximum_timestep=0.4,
    input_port_index=plant.get_actuation_input_port().get_index(),
)
prog = dircol.prog()

dircol.AddEqualTimeIntervalsConstraints()

initial_state = (0, 0.0, 0.0, np.pi, 0.0, 0.0, 0.0, 0.0)
prog.AddBoundingBoxConstraint(initial_state, initial_state, dircol.initial_state())
# More elegant version is blocked by drake #8315:
# prog.AddLinearConstraint(dircol.initial_state() == initial_state)

final_state = (0.0, np.pi, np.pi, 0.0, 0.0, 0.0, 0.0, 0.0)
prog.AddBoundingBoxConstraint(final_state, final_state, dircol.final_state())
# prog.AddLinearConstraint(dircol.final_state() == final_state)


R = 10  # Cost on input "effort".
u = dircol.input()
dircol.AddRunningCost(R * u[0] ** 2)

# Add a final cost equal to the total duration.
dircol.AddFinalCost(dircol.time())

# Providing the initial guess for x(.) as a straight line trajectory
# between the start and the goal
initial_x_trajectory = PiecewisePolynomial.FirstOrderHold(
    [0.0, 4.0], np.column_stack((initial_state, final_state))
)
dircol.SetInitialTrajectory(PiecewisePolynomial(), initial_x_trajectory)


# Add obstacles
x = dircol.state()
dircol.AddConstraintToAllKnotPoints(x[0] <= 2)
dircol.AddConstraintToAllKnotPoints(x[0] >= -2)

# Add obstacles
# for i in range(NUM_BREAKPOINTS*len(initial_state)):
#     if i % len(initial_state) == 0:
#         prog.AddLinearConstraint(x(i) <= 1)
#         print(i)
# prog.AddLinearConstraint()

# View the constraints that were added to the program
print("Details of prog:\n", print(prog))

result = Solve(prog)
assert result.is_success()

fig, ax = plt.subplots()

u_trajectory = dircol.ReconstructInputTrajectory(result)
times = np.linspace(u_trajectory.start_time(), u_trajectory.end_time(), 100)
u_lookup = np.vectorize(u_trajectory.value)
u_values = u_lookup(times)

ax.plot(times, u_values)
ax.set_xlabel("time (seconds)")
ax.set_ylabel("force (Newtons)")
display(plt.show())

# Animate the results.
x_trajectory = dircol.ReconstructStateTrajectory(result)

# TODO(russt): Add some helper methods to make this workflow cleaner.
builder = DiagramBuilder()
source = builder.AddSystem(TrajectorySource(x_trajectory))
builder.AddSystem(scene_graph)
pos_to_pose = builder.AddSystem(
    MultibodyPositionToGeometryPose(plant, input_multibody_state=True)
)
builder.Connect(source.get_output_port(0), pos_to_pose.get_input_port())
builder.Connect(
    pos_to_pose.get_output_port(),
    scene_graph.get_source_pose_port(plant.get_source_id()),
)

visualizer = builder.AddSystem(
    PlanarSceneGraphVisualizer(scene_graph, xlim=[-4, 4], ylim=[-2, 3], show=False)
)
builder.Connect(scene_graph.get_query_output_port(), visualizer.get_input_port(0))
simulator = Simulator(builder.Build())

AdvanceToAndVisualize(
    simulator, visualizer, x_trajectory.end_time() if running_as_notebook else 0.1
)

# %%
