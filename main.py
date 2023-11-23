from graphcast import graphcast
from graphcast import checkpoint
import numpy as np

SOURCE_1 = "random"
SOURCE_2 = "checkpoint"
SOURCE_TO_USE = SOURCE_2

# CONSTANTS FOR GRAPHCAST MODEL CONFIG IS SOURCE = RANDOM
RANDOM_MESH_SIZE = 5
RANDOM_GNN_MSG_STEPS = 10
RANDOM_LATENT_SIZE = 32
RANDOM_LEVELS = 13

# CREATING MODEL CONFIG
if SOURCE_TO_USE == SOURCE_1:
    params = None
    state = {}
    model_config = graphcast.ModelConfig(
        resolution=0,
        mesh_size=RANDOM_MESH_SIZE,
        latent_size=RANDOM_LATENT_SIZE,
        gnn_msg_steps=RANDOM_GNN_MSG_STEPS,
        hidden_layers=1,
        radius_query_fraction_edge_length=0.6)
    task_config = graphcast.TaskConfig(
        input_variables=graphcast.TASK.input_variables,
        target_variables=graphcast.TASK.target_variables,
        forcing_variables=graphcast.TASK.forcing_variables,
        pressure_levels=graphcast.PRESSURE_LEVELS[RANDOM_LEVELS],
        input_duration=graphcast.TASK.input_duration,
    )
else:
    # SOURCE SHOULD BE CHECKPOINT
    assert SOURCE_TO_USE == SOURCE_2
    # OPEN GRAPHCAST SMALL FILE PRESENT IN THE LOCAL DIRECTORY
    with np.load('./params_GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz') as data:
        print(data)