%load aimport

from src.models.svr_wrapper import ChainSVRWrapper, MultiSVRWrapper
from src.models.obn_wrapper import OBNWrapper
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
        [ChainSVRWrapper, {}],
        [MultiSVRWrapper, {"n_cpus" : 1}],
    ), 
    
    # GRID SEARCH PARAMS
    "restart" : False,
    "n_combis" : 1,
    "n_rep" : 1,
    "n_cpus" : 8,
    "fast" : True,
}
run(**kwargs)
