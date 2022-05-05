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
    LogVectorOutput,
    StartMeshcat,
    SceneGraph,
    MeshcatVisualizerCpp,
    RigidTransform,
    RotationMatrix,
)
import os
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from IPython.display import display, clear_output, SVG, HTML

# %%
def simulate_triple_cartpole(sim_time=10):

    # start construction site of our block diagram
    builder = DiagramBuilder()

    # instantiate the cart-pole and the scene graph
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    Parser(plant).AddModelFromFile("triple_cartpole.urdf")
    plant.Finalize()

    # set initial unstable equilibrium point
    context = plant.CreateDefaultContext()
    x_star = [0, np.pi, 0, 0, 0, 0, 0, 0]
    context.get_mutable_continuous_state_vector().SetFromVector(x_star)

    # weight matrices for the lqr controller
    Q = np.diag((10.0, 10.0, 10.0, 10.0, 1.0, 1.0, 1.0, 1.0))
    R = np.eye(1)

    # Setup input
    plant.get_actuation_input_port().FixValue(context, [0])
    input_i = plant.get_actuation_input_port().get_index()
    lqr = LinearQuadraticRegulator(plant, context, Q, R, input_port_index=int(input_i))
    lqr = builder.AddSystem(lqr)
    builder.Connect(plant.get_state_output_port(), lqr.get_input_port(0))
    builder.Connect(lqr.get_output_port(0), plant.get_actuation_input_port())

    # Add loggers
    state_logger = LogVectorOutput(plant.get_state_output_port(), builder)
    state_logger.set_name("state logger")

    input_logger = LogVectorOutput(lqr.get_output_port(), builder)
    input_logger.set_name("input logger")

    # Add visualizer
    visualizer = builder.AddSystem(
        PlanarSceneGraphVisualizer(
            scene_graph, xlim=[-4.0, 1.0], ylim=[-0.5, 3.2], show=False
        )
    )
    visualizer.set_name("visualizer")
    builder.Connect(scene_graph.get_query_output_port(), visualizer.get_input_port(0))

    # finish building the block diagram
    diagram = builder.Build()

    # instantiate a simulator
    simulator = Simulator(diagram)
    simulator.set_publish_every_time_step(False)  # makes sim faster

    context = simulator.get_mutable_context()
    context.SetTime(0)
    context.SetContinuousState(
        np.array([-2, 0.95 * np.pi, 0.05 * np.pi, 0.02 * np.pi, 0, 0, 0, 0])
    )

    # run simulation
    visualizer.start_recording()
    simulator.Initialize()
    simulator.set_target_realtime_rate(1.0)
    simulator.AdvanceTo(sim_time)

    # show visualization
    visualizer.stop_recording()
    ani = visualizer.get_recording_as_animation()
    display(HTML(ani.to_jshtml()))
    visualizer.reset_recording()

    # return the state and input over time
    state_log = state_logger.FindLog(simulator.get_context())
    input_log = input_logger.FindLog(simulator.get_context())
    state_names = list(
        [
            "x",
            "theta_1",
            "theta_2",
            "theta_3",
            "x_dot",
            "theta_1_dot",
            "theta_2_dot",
            "theta_3_dot",
        ]
    )
    df = pd.DataFrame(state_log.data().T, columns=state_names)
    df["time"] = state_log.sample_times()
    df["u"] = input_log.data().T
    return df


results = simulate_triple_cartpole()

# %%
# Create a plot of the showing the evolution of state variables over time
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 9))

sns.lineplot(data=results, x="time", y="x", ax=ax1)
ax1.set(ylabel="Cart Position (m)")

results["theta_1_minus_pi"] = results["theta_1"] - np.pi
sns.lineplot(data=results, x="time", y="theta_1_minus_pi", ax=ax2)
sns.lineplot(data=results, x="time", y="theta_2", ax=ax2)
sns.lineplot(data=results, x="time", y="theta_3", ax=ax2)
ax2.set(ylabel="Joint Angles (rad)")

sns.lineplot(data=results, x="time", y="theta_1_dot", ax=ax3)
sns.lineplot(data=results, x="time", y="theta_2_dot", ax=ax3)
sns.lineplot(data=results, x="time", y="theta_3_dot", ax=ax3)
ax3.set(ylabel="Joint Velocities (rad/sec)")

sns.lineplot(data=results, x="time", y="u", ax=ax4)
ax4.set(ylabel="Input (N)")

plt.show()
# %%

# %%
# An example of a phase portait which could be used for showing ROA slices
sns.lineplot(data=results, x="theta_1", y="theta_1_dot", sort=False)
# %%
