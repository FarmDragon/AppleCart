from pydrake.all import (
    DiagramBuilder,
    AddMultibodyPlantSceneGraph,
    Parser,
    FirstOrderTaylorApproximation,
)
import numpy as np


def linearize_cartpole(x0=[0, np.pi, 0, 0], u0=0):
    builder = DiagramBuilder()
    plant, _ = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    Parser(plant).AddModelFromFile("cartpole.urdf")
    plant.Finalize()
    context = plant.CreateDefaultContext()
    context.get_mutable_continuous_state_vector().SetFromVector(x0)
    plant.get_actuation_input_port().FixValue(context, [u0])
    input_port = plant.get_actuation_input_port().get_index()
    output_port = plant.get_state_output_port().get_index()
    return FirstOrderTaylorApproximation(plant, context, input_port, output_port)


# Example usage:
taylor_expansion = linearize_cartpole()

print(taylor_expansion.A())
print(taylor_expansion.B())

taylor_expansion = linearize_cartpole(x0=[0, np.pi * 0.2, 0, 0], u0=2)

print(taylor_expansion.A())
print(taylor_expansion.B())
