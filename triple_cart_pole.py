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
    RigidTransform,
    RotationMatrix,
)
import os
import numpy as np
from matplotlib import pyplot as plt


# %%
# Start the visualizer (run this cell only once, each instance consumes a port)
meshcat = StartMeshcat()

# %%
def simulate_triple_cartpole():

    # start construction site of our block diagram
    builder = DiagramBuilder()

    # instantiate the cart-pole and the scene graph
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    Parser(plant).AddModelFromFile("triple_cartpole.urdf")
    plant.Finalize()

    # set initial unstable equilibrium point
    context = plant.CreateDefaultContext()
    x_star = [0, np.pi, np.pi, 0, 0, 0, 0, 0]
    context.get_mutable_continuous_state_vector().SetFromVector(x_star)

    # weight matrices for the lqr controller
    Q = np.diag((10.0, 10.0, 10.0, 10.0, 1.0, 1.0, 1.0, 1.0))
    R = np.eye(1)

    # Setup input
    plant.get_actuation_input_port().FixValue(context, [0])
    input_i = plant.get_actuation_input_port().get_index()
    lqr = LinearQuadraticRegulator(plant, context, Q, R, input_port_index=int(input_i))
    lqr = builder.AddSystem(lqr)
    output_i = plant.get_state_output_port().get_index()
    cartpole_lin = Linearize(
        plant, context, input_port_index=input_i, output_port_index=output_i
    )
    builder.Connect(plant.get_state_output_port(), lqr.get_input_port(0))
    builder.Connect(lqr.get_output_port(0), plant.get_actuation_input_port())

    # Setup visualization
    MeshcatVisualizerCpp.AddToBuilder(builder, scene_graph, meshcat)
    meshcat.Delete()
    meshcat.Set2dRenderMode(xmin=-3, xmax=3, ymin=-1.0, ymax=4)

    # finish building the block diagram
    diagram = builder.Build()

    # instantiate a simulator
    simulator = Simulator(diagram)
    simulator.set_publish_every_time_step(False)  # makes sim faster

    context = simulator.get_mutable_context()
    context.SetTime(0)
    context.SetContinuousState(np.array([-2, 0.95*np.pi, 1.05*np.pi, .02*np.pi, 0, 0, 0, 0]))

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

state_traj = simulate_triple_cartpole()

# %%
# Create a plot of the showing the evolution of state variables over time
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(311)
plt.plot([state[0] for state in state_traj])
plt.title('x')

ax = fig.add_subplot(312)
plt.plot([state[1] for state in state_traj])
plt.plot([state[2] for state in state_traj])
# NOTE: Added pi to joint angle 3 so all angles are of comparable magnitude
plt.plot([state[3]+np.pi for state in state_traj])
plt.title('Joint Angles')

ax = fig.add_subplot(313)
plt.plot([state[5] for state in state_traj])
plt.plot([state[6] for state in state_traj])
plt.plot([state[7] for state in state_traj])
plt.title('Joint Velocities')
plt.show()



# %%
# An example of a phase portait which could be used for showing ROA slices
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot([state[1] for state in state_traj],[state[2] for state in state_traj])
ax.set_aspect('equal', adjustable='box')
plt.show()
# %%
