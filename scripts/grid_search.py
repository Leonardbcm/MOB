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

You can specify a number of cpus for parallelizing the configuration trainings. 
However, some models already natively supports parallelizing training.

ChainSVR works on single cpu, so you can parallelize configurations using all cpus
MultiSVR by default uses all cpus, so don't parallelize configurations
DNN and CNN training is not fully parallelized, so its possible to assign 
configurations for each cpu. However, they can eat memory so its better to use only
half the cpus for the configurations
OBN training uses all cpus, so use only 1 cpu for configuration.
"""
n_cpus = os.cpu_count()
half_cpus = int(n_cpus/2)
kwargs = {
    "GLOBAL_SEED" : 0,
    
    # TASKS
    "GRID_SEARCH" : True,

    # GENERAL PARAMS
    "n_val" : 365,
    "models" : (
        #[ChainSVRWrapper, {"n_cpus" : n_cpus}],
        #[MultiSVRWrapper, {"n_cpus" : 1}],
        #[DNNWrapper, {"n_cpus" : half_cpus}],        
        #[CNNWrapper, {"n_cpus" : half_cpus}],
        [OBNWrapper, {"n_cpus" : 1}],
    ), 
    
    # GRID SEARCH PARAMS
    "restart" : True,
    "n_combis" : 1,
    "n_rep" : 5000,
    "fast" : False,
}
run(**kwargs)
