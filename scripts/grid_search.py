import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3" 
#%load aimport

from src.models.svr_wrapper import ChainSVRWrapper, MultiSVRWrapper
from src.models.torch_wrapper import OBNWrapper
from src.models.keras_wrapper import DNNWrapper, CNNWrapper
from src.models.grid_search import run

"""
This script performs the hyper-parameter grid search for the SVR.
The sampling strategies and search spaces are defined each model wrapper's file.
This will overwrite files in data/Grid Search, copy them or set restart = FALSE.

We obtained our results using XXh per model on a XXcpu machine. 
To run it faster, lower the number of hyperparameter combinations to try 
'n_combi' or direclty re-use our grid search results.

Lower the number of cpus in case of high memory usage.
"""
kwargs = {
    # TASKS
    "GRID_SEARCH" : True,

    # GENERAL PARAMS
    "n_val" : 365,
    "models" : (
        #[ChainSVRWrapper, {}],
        #[MultiSVRWrapper, {"n_cpus" : 1}],
        [CNNWrapper, {"n_cpus" : os.cpu_count()}],
    ), 
    
    # GRID SEARCH PARAMS
    "restart" : False,
    "n_combis" : 2,
    "n_rep" : 5,
    "fast" : True,
}
run(**kwargs)
