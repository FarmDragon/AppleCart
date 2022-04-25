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
import webbrowser

class Cart:

    def __init__(self) -> None:
        self.x = [0,0,0,0,0,0,0,0] # state of the cart
        self.meshcat = StartMeshcat()
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
        Q = np.diag((10., 10., 10., 10., 1., 1., 1., 1.))
        R = np.eye(1)

        # Setup input
        plant.get_actuation_input_port().FixValue(context, [0])
        input_i = plant.get_actuation_input_port().get_index()
        lqr = LinearQuadraticRegulator(plant, context, Q, R, input_port_index=int(input_i))
        lqr = builder.AddSystem(lqr)
        output_i = plant.get_state_output_port().get_index()
        cartpole_lin = Linearize(plant,
                                context,
                                input_port_index=input_i,
                                output_port_index=output_i)
        builder.Connect(plant.get_state_output_port(), lqr.get_input_port(0))
        builder.Connect(lqr.get_output_port(0), plant.get_actuation_input_port())

        Cart.__setup_visualization(builder, scene_graph, self.meshcat)

        self.diagram = builder.Build()

        # instantiate a simulator
        simulator = Simulator(self.diagram)
        simulator.set_publish_every_time_step(False)  # makes sim faster
        self.simulator = simulator

        context = simulator.get_mutable_context()
        context.SetTime(0)
        context.SetContinuousState(np.array([-2, 0.95*np.pi, 1.05*np.pi, .01*np.pi, 0, 0, 0, 0]))
    
    @staticmethod
    def __setup_visualization(builder, scene_graph, meshcat):
        MeshcatVisualizerCpp.AddToBuilder(builder, scene_graph, meshcat)
        meshcat.Delete()
        meshcat.Set2dRenderMode(xmin=-5, xmax=5, ymin=-5.0, ymax=5)

    def start_simulation(self):
        # run simulation
        self.meshcat.AddButton("Stop Simulation")
        self.simulator.Initialize()
        self.simulator.set_target_realtime_rate(1.0)
        webbrowser.open(self.meshcat.web_url())
        while self.meshcat.GetButtonClicks("Stop Simulation") < 1:
            if self.simulator.get_context().get_time() > 20:
                break
            self.simulator.AdvanceTo(self.simulator.get_context().get_time() + 1.0)
        self.meshcat.DeleteAddedControls()