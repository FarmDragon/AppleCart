# %%
import numpy as np
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    MultilayerPerceptron,
    Parser,
    PerceptronActivationType,
    Simulator,
    PlanarSceneGraphVisualizer,
    ZeroOrderHold,
    # FirstOrderHold,
    FittedValueIteration,
    DynamicProgrammingOptions,
    SceneGraph,
    LogVectorOutput,
    WrapToSystem,
    LinearQuadraticRegulator,
    PiecewisePolynomial,
    TrajectorySource,
    FiniteHorizonLinearQuadraticRegulatorOptions,
    MakeFiniteHorizonLinearQuadraticRegulator,
)
from IPython.display import HTML, display
from IPython.display import display, clear_output, SVG, display
import pydot
import pandas as pd
import altair as alt


# %%
def visualize_system(system):
    display(
        SVG(
            pydot.graph_from_dot_data(system.GetGraphvizString(max_depth=2))[
                0
            ].create_svg()
        )
    )


def run_trajectory(
    inputs,
    states=None,
    sim_time=4,
    starting_state={
        "x": 0,
        "theta_1": 0,
        "theta_2": 0,
        "theta_3": 0,
        "x_dot": 0,
        "theta_dot_1": 0,
        "theta_dot_2": 0,
        "theta_dot_3": 0,
    },
    use_lqr=False,
):
    builder = DiagramBuilder()

    # Create the cartpole
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    Parser(plant).AddModelFromFile("../triple_cartpole.urdf")
    plant.Finalize()
    plant_context = plant.CreateDefaultContext()
    plant.set_name("cartpole")

    # Add the controller
    times = np.linspace(0, sim_time, num=len(inputs))
    input_polynomial = PiecewisePolynomial.FirstOrderHold(
        times, inputs.reshape(-1, 1).T
    )

    if use_lqr:
        assert states is not None
        Q = np.diag([10.0, 10.0, 10.0, 10.0, 1.0, 1.0, 1.0, 1.0])
        R = 0.001 * np.eye(1)
        options = FiniteHorizonLinearQuadraticRegulatorOptions()
        options.Qf = np.diag([1000.0, 1000.0, 1000.0, 1000.0, 1.0, 1.0, 1.0, 1.0])
        state_polynomial = PiecewisePolynomial.FirstOrderHold(
            times, states.reshape(-1, 1).T
        )
        options.x0 = state_polynomial
        options.u0 = input_polynomial
        options.input_port_index = plant.get_actuation_input_port().get_index()

        controller = builder.AddSystem(
            MakeFiniteHorizonLinearQuadraticRegulator(
                system=plant,
                context=plant_context,
                t0=state_polynomial.start_time(),
                tf=state_polynomial.end_time(),
                Q=Q,
                R=R,
                options=options,
            )
        )
        controller.set_name("feedback controller")
        builder.Connect(controller.get_output_port(), plant.get_actuation_input_port())
        builder.Connect(plant.get_state_output_port(), controller.get_input_port())
    else:
        controller = builder.AddSystem(TrajectorySource(input_polynomial))
        controller.set_name("feed-forward controller")
        builder.Connect(controller.get_output_port(), plant.get_actuation_input_port())

    # Add visualizer
    visualizer = builder.AddSystem(
        PlanarSceneGraphVisualizer(
            scene_graph, xlim=[-3.0, 3.0], ylim=[-4, 4], show=False
        )
    )
    visualizer.set_name("visualizer")
    builder.Connect(scene_graph.get_query_output_port(), visualizer.get_input_port(0))

    # Add loggers
    state_logger = LogVectorOutput(plant.get_state_output_port(), builder)
    state_logger.set_name("state logger")

    input_logger = LogVectorOutput(controller.get_output_port(), builder)
    input_logger.set_name("input logger")

    # Build the diagram
    diagram = builder.Build()
    diagram.set_name("diagram")
    visualize_system(diagram)

    # Initialize simulator
    simulator = Simulator(diagram)
    simulator.set_publish_every_time_step(False)  # makes sim faster
    simulator.set_target_realtime_rate(1.0)

    # Simulate
    visualizer.start_recording()

    context = simulator.get_mutable_context()
    context.SetTime(0.0)
    context.SetContinuousState(np.array(list(starting_state.values())))

    state_logger.FindMutableLog(context).Clear()
    input_logger.FindMutableLog(context).Clear()

    simulator.Initialize()
    simulator.AdvanceTo(sim_time)

    visualizer.stop_recording()

    ani = visualizer.get_recording_as_animation()
    display(HTML(ani.to_jshtml()))
    visualizer.reset_recording()

    state_log = state_logger.FindLog(simulator.get_context())
    input_log = input_logger.FindLog(simulator.get_context())

    state_names = list(starting_state.keys())
    df = pd.DataFrame(state_log.data().T, columns=state_names)
    df["time"] = state_log.sample_times()
    df["u"] = input_log.data().T

    return df


# %%
df = run_trajectory(np.load("trajectory_inputs.npy"))

# %%
base_chart = alt.Chart(df).encode(
    alt.X("time:Q", title="Time (s)"),
)
input_chart = base_chart.mark_line(color="orange").encode(
    alt.Y("u:Q", title="Torque (Nm)", axis=alt.Axis(titleColor="orange")),
)
theta_1_chart = base_chart.mark_line().encode(
    alt.Y("theta_1:Q", title="Theta (rad)"),
)

alt.layer(input_chart, theta_1_chart).resolve_scale(y="independent")

# %%
