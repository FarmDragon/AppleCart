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
        self.meshCat = StartMeshcat()
        builder = DiagramBuilder()

        # instantiate the cart-pole and the scene graph
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
        Parser(plant).AddModelFromFile("triple_cartpole.urdf")
        plant.Finalize()

        # set initial unstable equilibrium point
        self.context = plant.CreateDefaultContext()
        x_star = [0, np.pi, np.pi, 0, 0, 0, 0, 0]
        self.context.get_mutable_continuous_state_vector().SetFromVector(x_star)

        Cart.__setup_visualization(builder, scene_graph, self.meshCat)

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
        while self.meshcat.GetButtonClicks("Stop Simulation") < 1:
            if self.simulator.get_context().get_time() > 20:
                break
            self.simulator.AdvanceTo(self.simulator.get_context().get_time() + 1.0)
        self.meshcat.DeleteAddedControls()
        webbrowser.open(f"localhost:{self.meshCat}")