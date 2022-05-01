# %% [markdown]
# # Authoring a Multibody Simulation
#
# This tutorial provides some tools to help you create a new scene description file that can be parsed into Drake's multibody physics engine (MultibodyPlant) and geometry engine (SceneGraph).
#
# You can **duplicate this notebook**, and edit your file directly here in deepnote.  Just create a new file, or upload and existing file along with your art assets (e.g. obj and mtl files) to this project.

# %%
# Imports
import numpy as np
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    FindResourceOrThrow,
    MeshcatVisualizerCpp,
    MeshcatVisualizerParams,
    Parser,
    RigidTransform,
    Role,
    RollPitchYaw,
    Simulator,
    StartMeshcat,
    JointSliders,
)

# %%
# Start the visualizer.
meshcat = StartMeshcat()

# %% [markdown]
# ## Inspecting a URDF / SDF using joint sliders
#
# The most important formats for creating multibody scenarios in Drake are the "Universal Robot Description Format" (URDF) and the "Scene Description Format" (SDF)...

# %%
def inspector(filename, package_paths={}):
    meshcat.Delete()
    meshcat.DeleteAddedControls()
    builder = DiagramBuilder()

    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)

    # Load the file into the plant/scene_graph.
    parser = Parser(plant)
    for name, directory in package_paths.items():
        parser.package_map().Add(name, directory)
    parser.AddModelFromFile(filename)
    plant.Finalize()

    # Add two visualizers, one to publish the "visual" geometry, and one to
    # publish the "collision" geometry.
    visual = MeshcatVisualizerCpp.AddToBuilder(
        builder,
        scene_graph,
        meshcat,
        MeshcatVisualizerParams(role=Role.kPerception, prefix="visual"),
    )
    collision = MeshcatVisualizerCpp.AddToBuilder(
        builder,
        scene_graph,
        meshcat,
        MeshcatVisualizerParams(role=Role.kProximity, prefix="collision"),
    )
    # Disable the collision geometry at the start; it can be enabled by the
    # checkbox in the meshcat controls.
    meshcat.SetProperty("collision", "visible", False)

    sliders = builder.AddSystem(JointSliders(meshcat, plant))

    diagram = builder.Build()
    sliders.Run(diagram)


inspector("triple_cartpole.urdf")
