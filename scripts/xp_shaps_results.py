%load aimport
import itertools, pandas, numpy as np, datetime, matplotlib.pyplot as plt, time
import matplotlib

from src.models.spliter import MySpliter
from src.models.torch_wrapper import OBNWrapper
from src.models.torch_models.weight_initializers import *
import src.models.parallel_scikit as ps
from src.analysis.utils import *
from src.analysis.xp_results_utils import *

"""
XP results and analysis file
"""

###### Load shap values
N_VAL = PARAMS["N_VAL"]
N_SAMPLES = PARAMS["N_SAMPLES"]
folder = PARAMS["tboard"]

matrix = np.zeros((n, N_GROUPS))
##### LOOP
ID = 1
OBs = 20
version = None

# Load raw shaps    
shaps = np.load(model_wrapper.version_shap_path(version_))

# Keep only contributions towards the prices
shaps = shaps[model_wrapper.y_indices, :, :]

# Sum across all prices
shaps = shaps.sum(axis=0)

# Keep only contributions from DATA (remove OB)
shaps = shaps[:, model_wrapper.x_indices]

# Unitarize shaps : take absolute values and sum must be 1 for all samples
shaps = np.abs(shaps) / np.abs(shaps).sum(axis=1).reshape(N_SAMPLES, 1)

# Group by variables
N_GROUPS = len(model_wrapper.variables)
grouped_shaps = np.zeros((N_SAMPLES, N_GROUPS))
for i, variable in enumerate(model_wrapper.variables):
    inds_variable = model_wrapper.get_variable_indices(variable)
    grouped_shaps[:, i] = shaps[:, inds_variable].sum(axis=1)

# Average across all samples
grouped_shaps = grouped_shaps.mean(axis=0)
