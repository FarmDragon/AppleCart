# %% [markdown]
# # Triple cart-pole

# %%
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    LinearQuadraticRegulator,
    Parser,
    PlanarSceneGraphVisualizer,
    Simulator,
    Linearize,
    StartMeshcat,
    SceneGraph,
    MeshcatVisualizerCpp,
    Solve,
    TrajectorySource,
    MultibodyPositionToGeometryPose,
    PiecewisePolynomial,
    DirectCollocation
    
)
import os
import numpy as np
from matplotlib import pyplot as plt


# %%
# Start the visualizer (run this cell only once, each instance consumes a port)
meshcat = StartMeshcat()

# %%
# def simulate_triple_cartpole():

#     # start construction site of our block diagram
#     builder = DiagramBuilder()

#     # instantiate the cart-pole and the scene graph
#     plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
#     Parser(plant).AddModelFromFile("triple_cartpole.urdf")
#     plant.Finalize()

#     # set initial unstable equilibrium point
#     context = plant.CreateDefaultContext()
#     x_star = [0, np.pi, np.pi, 0, 0, 0, 0, 0]
#     context.get_mutable_continuous_state_vector().SetFromVector(x_star)

#     # weight matrices for the lqr controller
#     Q = np.diag((10., 10., 10., 10., 1., 1., 1., 1.))
#     R = np.eye(1)

#     # Setup input
#     plant.get_actuation_input_port().FixValue(context, [0])
#     input_i = plant.get_actuation_input_port().get_index()
#     lqr = LinearQuadraticRegulator(plant, context, Q, R, input_port_index=int(input_i))
#     lqr = builder.AddSystem(lqr)
#     output_i = plant.get_state_output_port().get_index()
#     cartpole_lin = Linearize(plant,
#                             context,
#                             input_port_index=input_i,
#                             output_port_index=output_i)
#     builder.Connect(plant.get_state_output_port(), lqr.get_input_port(0))
#     builder.Connect(lqr.get_output_port(0), plant.get_actuation_input_port())

#     # Setup visualization
#     MeshcatVisualizerCpp.AddToBuilder(builder, scene_graph, meshcat)
#     meshcat.Delete()
#     meshcat.Set2dRenderMode(xmin=-3, xmax=3, ymin=-1.0, ymax=4)

#     # finish building the block diagram
#     diagram = builder.Build()

#     # instantiate a simulator
#     simulator = Simulator(diagram)
#     simulator.set_publish_every_time_step(False)  # makes sim faster

#     context = simulator.get_mutable_context()
#     context.SetTime(0)
#     context.SetContinuousState(np.array([-2, 0.95*np.pi, 1.05*np.pi, .02*np.pi, 0, 0, 0, 0]))

#     # run simulation
#     meshcat.AddButton("Stop Simulation")
#     simulator.Initialize()
#     simulator.set_target_realtime_rate(1.0)
    
#     state_traj = []
#     input("Press Enter to start simulation")
    
#     while meshcat.GetButtonClicks("Stop Simulation") < 1:
#         print('Time:', simulator.get_context().get_time())
#         x = simulator.get_context().get_continuous_state().get_generalized_position().GetAtIndex(0)
        
#         state = simulator.get_context().get_continuous_state().get_vector().CopyToVector()
#         state_traj.append(state)
#         if simulator.get_context().get_time() >= 10:
#             break
#         simulator.AdvanceTo(simulator.get_context().get_time() + 0.01)
#     meshcat.DeleteAddedControls()
    
#     return state_traj

# state_traj = simulate_triple_cartpole()

# %%
# Create a plot of the showing the evolution of state variables over time
# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(311)
# plt.plot([state[0] for state in state_traj])
# plt.title('x')

# ax = fig.add_subplot(312)
# plt.plot([state[1] for state in state_traj])
# plt.plot([state[2] for state in state_traj])
# # NOTE: Added pi to joint angle 3 so all angles are of comparable magnitude
# plt.plot([state[3]+np.pi for state in state_traj])
# plt.title('Joint Angles')

# ax = fig.add_subplot(313)
# plt.plot([state[5] for state in state_traj])
# plt.plot([state[6] for state in state_traj])
# plt.plot([state[7] for state in state_traj])
# plt.title('Joint Velocities')
# plt.show()



# %%
# An example of a phase portait which could be used for showing ROA slices
# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.plot([state[1] for state in state_traj],[state[2] for state in state_traj])
# ax.set_aspect('equal', adjustable='box')
# plt.show()
# %%

def dircol_cartpole():
    # start construction site of our block diagram
    builder = DiagramBuilder()

    # instantiate the cart-pole and the scene graph
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    Parser(plant).AddModelFromFile("triple_cartpole.urdf")
    plant.Finalize()
    
    scene_graph = SceneGraph()

    context = plant.CreateDefaultContext()
    dircol = DirectCollocation(
        plant,
        context,
        num_time_samples=21,
        minimum_timestep=0.1,
        maximum_timestep=0.4,
        input_port_index=plant.get_actuation_input_port().get_index())
    prog = dircol.prog()

    dircol.AddEqualTimeIntervalsConstraints()

    initial_state = (0., 0., 0., 0., 0., 0.)
    prog.AddBoundingBoxConstraint(initial_state, initial_state,
                                  dircol.initial_state())
    # More elegant version is blocked by drake #8315:
    # prog.AddLinearConstraint(dircol.initial_state() == initial_state)

    final_state = (0, np.pi, np.pi, 0, 0, 0, 0, 0)
    prog.AddBoundingBoxConstraint(final_state, final_state,
                                  dircol.final_state())
    # prog.AddLinearConstraint(dircol.final_state() == final_state)

    R = 10  # Cost on input "effort".
    u = dircol.input()
    dircol.AddRunningCost(R * u[0]**2)

    # Add a final cost equal to the total duration.
    dircol.AddFinalCost(dircol.time())

    initial_x_trajectory = PiecewisePolynomial.FirstOrderHold(
        [0., 4.], np.column_stack((initial_state, final_state)))  # yapf: disable
    dircol.SetInitialTrajectory(PiecewisePolynomial(), initial_x_trajectory)

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
    plt.show()

    # Animate the results.
    x_trajectory = dircol.ReconstructStateTrajectory(result)

    # TODO(russt): Add some helper methods to make this workflow cleaner.
    builder = DiagramBuilder()
    source = builder.AddSystem(TrajectorySource(x_trajectory))
    builder.AddSystem(scene_graph)
    pos_to_pose = builder.AddSystem(
        MultibodyPositionToGeometryPose(plant, input_multibody_state=True))
    builder.Connect(source.get_output_port(0), pos_to_pose.get_input_port())
    builder.Connect(pos_to_pose.get_output_port(),
                    scene_graph.get_source_pose_port(plant.get_source_id()))

    visualizer = builder.AddSystem(
        PlanarSceneGraphVisualizer(scene_graph,
                                   xlim=[-2, 2],
                                   ylim=[-1.25, 2],
                                   show=False))
    builder.Connect(scene_graph.get_query_output_port(),
                    visualizer.get_input_port(0))
    simulator = Simulator(builder.Build())

    # AdvanceToAndVisualize(
    #     simulator, visualizer,
    #     x_trajectory.end_time() if running_as_notebook else 0.1)
    
    # run simulation
    meshcat.AddButton("Stop Simulation")
    simulator.Initialize()
    simulator.set_target_realtime_rate(1.0)
    
    state_traj = []
    input("Press Enter to start simulation")
    
    while meshcat.GetButtonClicks("Stop Simulation") < 1:
        print('Time:', simulator.get_context().get_time())
        x = simulator.get_context().get_continuous_state().get_generalized_position().GetAtIndex(0)
        
        state = simulator.get_context().get_continuous_state().get_vector().CopyToVector()
        state_traj.append(state)
        if simulator.get_context().get_time() >= 10:
            break
        simulator.AdvanceTo(simulator.get_context().get_time() + 0.01)
    meshcat.DeleteAddedControls()
    
    return state_traj

state_traj = dircol_cartpole()

# %%
