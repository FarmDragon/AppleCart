# %% [markdown]
# # Triple cart-pole

# %%
# Import necessary packages
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
    Cylinder,
    StartMeshcat,
    TrajectorySource,
    Variable,
    eq,
    MeshcatVisualizerCpp,
    AddMultibodyPlantSceneGraph,
    MultibodyForces_,
)
from pydrake.examples.acrobot import AcrobotGeometry, AcrobotPlant
from pydrake.examples.pendulum import PendulumPlant, PendulumState

from pydrake.systems.framework import BasicVector, LeafSystem, PortDataType

from underactuated import FindResource, running_as_notebook
from underactuated.jupyter import AdvanceToAndVisualize
from underactuated.meshcat_utils import draw_points, set_planar_viewpoint
from underactuated.pendulum import PendulumVisualizer

# %%
# Start Meshcat and load model and constances
# Start the visualizer (run this cell only once, each instance consumes a port)
meshcat = StartMeshcat()

# Triple cart pole model
TRIPLE_CARTPOLE_URDF = "../triple_cartpole.urdf"
SIMULATION_TIMESTEP = 0.01
FLAT_SIMULATION = True
MESHCAT_2D_LIMITS = {"xmin": -5, "xmax": 3, "ymin": -3.5, "ymax": 3.5}
MAX_SIMULATION_TIME = 6  # seconds after which to stop meshcat simulation

# %%
# Define helper functions
def draw_simulation_env(meshcat, FLAT_SIMULATION):
    meshcat.SetObject("apple", Sphere(0.1), Rgba(1, 0, 0, 1))
    meshcat.SetTransform("apple", RigidTransform([2, 0, 3]))

    meshcat.SetObject("branch", Cylinder(0.6, 1), Rgba(0.5, 0.4, 0.3, 1))

    R_GgraspO = RotationMatrix.MakeXRotation(np.pi / 2.0).multiply(
        RotationMatrix.MakeZRotation(np.pi / 2.0)
    )
    meshcat.SetTransform("branch", RigidTransform(R_GgraspO, [0, 0, 1.5]))

    # Visualize the obstacles
    meshcat.SetObject("wall1", Box(1, 1, 1), Rgba(0.8, 0.4, 0, 1))
    meshcat.SetTransform("wall1", RigidTransform([-5, 0, 0]))

    meshcat.SetObject("wall2", Box(1, 1, 1), Rgba(0.8, 0.4, 0, 1))
    meshcat.SetTransform("wall2", RigidTransform([5, 0, 0]))

    if FLAT_SIMULATION:
        meshcat.Set2dRenderMode(
            xmin=MESHCAT_2D_LIMITS["xmin"],
            xmax=MESHCAT_2D_LIMITS["xmax"],
            ymin=MESHCAT_2D_LIMITS["ymin"],
            ymax=MESHCAT_2D_LIMITS["ymax"],
        )


# %%
# Solve trajectory optimization

# NOTE: These are the original values I tried
# NUM_BREAKPOINTS = 21
# MIN_TIMESTEP = 0.1
# MAX_TIMESTEP = 0.4

# NOTE: One suprising thing is how different the force profiles are when these values are changed even slightly
# NUM_BREAKPOINTS = 81
# MIN_TIMESTEP = 0.025
# MAX_TIMESTEP = 0.1

# NOTE: These parameters take a while, but the input trajectory is smoother
NUM_BREAKPOINTS = 191
MIN_TIMESTEP = 0.00175
MAX_TIMESTEP = 0.025

plant = MultibodyPlant(time_step=0.0)
scene_graph = SceneGraph()
plant.RegisterAsSourceForSceneGraph(scene_graph)
Parser(plant).AddModelFromFile(TRIPLE_CARTPOLE_URDF)
plant.Finalize()

context = plant.CreateDefaultContext()
dircol = DirectCollocation(
    plant,
    context,
    num_time_samples=NUM_BREAKPOINTS,
    minimum_timestep=MIN_TIMESTEP,
    maximum_timestep=MAX_TIMESTEP,
    input_port_index=plant.get_actuation_input_port().get_index(),
)
prog = dircol.prog()

dircol.AddEqualTimeIntervalsConstraints()

initial_state = [-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
prog.AddBoundingBoxConstraint(initial_state, initial_state, dircol.initial_state())
# More elegant version is blocked by drake #8315:
# prog.AddLinearConstraint(dircol.initial_state() == initial_state)

final_state = (2, np.pi, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
prog.AddBoundingBoxConstraint(final_state, final_state, dircol.final_state())
# prog.AddLinearConstraint(dircol.final_state() == final_state)

# R = 10
R = 1000  # Cost on input "effort".
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
dircol.AddConstraintToAllKnotPoints(x[0] <= 4)
dircol.AddConstraintToAllKnotPoints(x[0] >= -4)


# Obstacle location
obs_x = 0
obs_y = 1.5
obs_radius = 0.6

# First constraint to avoid branch
x1 = x[0] + np.sin(x[1])
y1 = -np.cos(x[1])
pendulum1_to_obstacle_squared = np.sqrt((obs_x - x1) ** 2 + (obs_y - y1) ** 2)
dircol.AddConstraintToAllKnotPoints(pendulum1_to_obstacle_squared >= obs_radius)

# Second constraint to avoid branch
x2 = x[0] + np.sin(x[1]) + np.sin(x[1] + x[2])
y2 = -np.cos(x[1]) - np.cos(x[1] + x[2])
pendulum2_to_obstacle_squared = np.sqrt((obs_x - x2) ** 2 + (obs_y - y2) ** 2)
dircol.AddConstraintToAllKnotPoints(pendulum2_to_obstacle_squared >= obs_radius)

# Third constraint to avoid branch
x3 = x[0] + np.sin(x[1]) + np.sin(x[1] + x[2]) + np.sin(x[1] + x[2] + x[3])
y3 = -np.cos(x[1]) - np.cos(x[1] + x[2]) - np.cos(x[1] + x[2] + x[3])
pendulum3_to_obstacle_squared = np.sqrt((obs_x - x3) ** 2 + (obs_y - y3) ** 2)
dircol.AddConstraintToAllKnotPoints(pendulum3_to_obstacle_squared >= obs_radius)


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
ax.set_xlabel("Time (s)")
ax.set_ylabel("Force (N)")
display(plt.show())

# Save the trajectory
x_trajectory = dircol.ReconstructStateTrajectory(result)


# %%
# Prepare the simulation of trajectory optimization

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

MeshcatVisualizerCpp.AddToBuilder(builder, scene_graph, meshcat)
meshcat.Delete()

if FLAT_SIMULATION:
    meshcat.Set2dRenderMode(
        xmin=MESHCAT_2D_LIMITS["xmin"],
        xmax=MESHCAT_2D_LIMITS["xmax"],
        ymin=MESHCAT_2D_LIMITS["ymin"],
        ymax=MESHCAT_2D_LIMITS["ymax"],
    )

diagram = builder.Build()


simulator = Simulator(diagram)
simulator.set_publish_every_time_step(False)  # makes sim faster

context = simulator.get_mutable_context()
# %%
# % Run the simulation of trajectory optimization
context.SetTime(0)

# run simulation
meshcat.AddButton("Stop Simulation")
draw_simulation_env(meshcat, FLAT_SIMULATION)


simulator.Initialize()
simulator.set_target_realtime_rate(1.0)

# state_traj = []
input("Press Enter to start simulation after 2 seconds")
time.sleep(2)  # Give time to switch to meshcat window

while meshcat.GetButtonClicks("Stop Simulation") < 1:
    print("Time:", simulator.get_context().get_time())
    # x = simulator.get_context().get_continuous_state().get_generalized_position().GetAtIndex(0)

    state = simulator.get_context().get_continuous_state().get_vector().CopyToVector()
    # print(state)
    # state_traj.append(state)
    if simulator.get_context().get_time() >= 6:
        break
    # Advance the simulation forward
    simulator.AdvanceTo(simulator.get_context().get_time() + SIMULATION_TIMESTEP)
    # Make the meshcat animation match
    time.sleep(SIMULATION_TIMESTEP)

meshcat.DeleteAddedControls()

# %%
# Run open loop simulation of trajectory optimization result
builder = DiagramBuilder()

plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
Parser(plant).AddModelFromFile(TRIPLE_CARTPOLE_URDF)
plant.Finalize()

visualizer = MeshcatVisualizerCpp.AddToBuilder(builder, scene_graph, meshcat)
logger = LogVectorOutput(plant.get_state_output_port(), builder)
meshcat.Delete()

traj = builder.AddSystem(TrajectorySource(u_trajectory))
builder.Connect(traj.get_output_port(), plant.get_actuation_input_port())
diagram = builder.Build()

simulator = Simulator(diagram)
context = simulator.get_mutable_context()
plant_context = plant.GetMyContextFromRoot(context)

ts = np.linspace(u_trajectory.start_time(), u_trajectory.end_time(), 301)
desired_state = x_trajectory.vector_values(ts)

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(ts, desired_state[0], label="desired")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Cart Position (m)")

context.SetTime(x_trajectory.start_time())
initial_state = x_trajectory.value(x_trajectory.start_time())
plant_context.SetContinuousState(initial_state[:])

draw_simulation_env(meshcat, FLAT_SIMULATION)

input("Press Enter to start simulation after 2 seconds")
time.sleep(2)  # Give time to switch to meshcat window

visualizer.StartRecording(False)

simulator.AdvanceTo(x_trajectory.end_time())
visualizer.PublishRecording()

log = logger.FindLog(context)
state = log.data()

ax.plot(log.sample_times(), state[0], label=f"actual")
ax.legend()
plt.show()
# %%
x_trajectory.vector_values(ts)
# %%
# Finite Horizon LQR to stabilze trajectory

builder = DiagramBuilder()

plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
Parser(plant).AddModelFromFile(TRIPLE_CARTPOLE_URDF)
plant.Finalize()

visualizer = MeshcatVisualizerCpp.AddToBuilder(builder, scene_graph, meshcat)
logger = LogVectorOutput(plant.get_state_output_port(), builder)
meshcat.Delete()

Q = np.diag([10.0, 10.0, 10.0, 10.0, 1.0, 1.0, 1.0, 1.0])
# NOTE: Tried reducing R so that there is not a great cost on input
R = 0.001 * np.eye(1)
options = FiniteHorizonLinearQuadraticRegulatorOptions()
# NOTE: Tried removing this final cost on the state
# NOTE: Tried making these values much bigger to penalize not reaching final state
options.Qf = np.diag([1000.0, 1000.0, 1000.0, 1000.0, 1.0, 1.0, 1.0, 1.0])
# options.use_square_root_method = True  # Pending drake PR #16812
options.x0 = x_trajectory
options.u0 = u_trajectory
options.input_port_index = plant.get_actuation_input_port().get_index()

# NOTE: The following line takes a long time
controller = builder.AddSystem(
    MakeFiniteHorizonLinearQuadraticRegulator(
        system=plant,
        context=plant.CreateDefaultContext(),
        t0=x_trajectory.start_time(),
        tf=x_trajectory.end_time(),
        Q=Q,
        R=R,
        options=options,
    )
)
builder.Connect(controller.get_output_port(), plant.get_actuation_input_port())
builder.Connect(plant.get_state_output_port(), controller.get_input_port())

diagram = builder.Build()

simulator = Simulator(diagram)
context = simulator.get_mutable_context()
plant_context = plant.GetMyContextFromRoot(context)

ts = np.linspace(u_trajectory.start_time(), u_trajectory.end_time(), 301)
# We plot this after running the simulation
desired_state = x_trajectory.vector_values(ts)

# %%

# Simulation the time varying LQR response (not recorded)
# This is a version that you can step through in the debugger
# The next cell advances to the end and records it
meshcat.AddButton("Stop Simulation")
rng = np.random.default_rng(123)

# for i in range(1):
context.SetTime(x_trajectory.start_time())
# initial_state = GliderState(x_traj.value(x_traj.start_time()))
initial_state = x_trajectory.value(x_trajectory.start_time())
# TODO: Add random noise to initial condition
initial_state[0] += 0.4 * rng.standard_normal()
plant_context.SetContinuousState(initial_state[:])

simulator.Initialize()
simulator.set_target_realtime_rate(1.0)

draw_simulation_env(meshcat, FLAT_SIMULATION)

# state_traj = []
input("Press Enter to start simulation after 2 seconds")
time.sleep(2)  # Give time to switch to meshcat window

count = 0

while meshcat.GetButtonClicks("Stop Simulation") < 1:
    print("Time:", simulator.get_context().get_time())
    # x = simulator.get_context().get_continuous_state().get_generalized_position().GetAtIndex(0)

    state = simulator.get_context().get_continuous_state().get_vector().CopyToVector()

    # NOTE: Adding a disturbance in the middle of the simulation
    # if count == 5:
    #     state[0] = 0
    #     plant_context.SetContinuousState(state)

    # print(state)
    # state_traj.append(state)
    if simulator.get_context().get_time() >= MAX_SIMULATION_TIME:
        break
    # Advance the simulation forward
    simulator.AdvanceTo(simulator.get_context().get_time() + SIMULATION_TIMESTEP)
    # Make the meshcat animation match
    time.sleep(SIMULATION_TIMESTEP)
    count += 1

meshcat.DeleteAddedControls()

# %%
# Simulation the time varying LQR response
rng = np.random.default_rng(123)


# for i in range(1):
context.SetTime(x_trajectory.start_time())
# initial_state = GliderState(x_traj.value(x_traj.start_time()))
initial_state = x_trajectory.value(x_trajectory.start_time())
# TODO: Add random noise to initial condition
initial_state[0] += 0.4 * rng.standard_normal()
plant_context.SetContinuousState(initial_state[:])

draw_simulation_env(meshcat, FLAT_SIMULATION)

input("Press Enter to start simulation after 2 seconds")
time.sleep(2)  # Give time to switch to meshcat window

logger.FindMutableLog(context).Clear()

simulator.Initialize()
simulator.AdvanceTo(x_trajectory.end_time())
# log.Clear()
visualizer.PublishRecording()
log = logger.FindLog(context)
state = log.data()

# %%
# Plot the results of the time varying LQR

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(ts, desired_state[0], label="desired")
ax.set_xlabel("time")
ax.set_ylabel("cart position")

ax.plot(log.sample_times(), state[0], "r--", label=f"actual")

ax.legend()
# In case the result blows up, at least see how close it was
# ax.set_ylim([-1, 0.5])
plt.show()
# %%
