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

####### MAIN PARAMS
N_EPOCHS = 1000
N_SAMPLES = 1444
N_VAL = 365
BATCH = 80

####### configurations
#countries = ["FR", "DE", "BE", "NL"]
countries = ["BE"]
#datasets = ["Lyon", "Munich", "Bruges", "Lahaye"]
datasets = ["Bruges"]
IDs = np.array([1, 2, 3, 4, 5, 6, 7])
OB_sizes = np.array([20, 50])

###### Retrieve several OBs
predicted_prices, real_prices, predicted_OB, real_OB, results=retrieve_results_OBs(
    IDs, countries, datasets, OB_sizes, N_VAL, N_SAMPLES, nh = 24)

key_order = ["country", "ID", "val_OB_smape", "val_OB_ACC", "val_price_mae",
             "val_price_smape", "val_price_ACC"]

res = results["50"]
res = res.sort_values(["country", "ID"]).loc[:, key_order]
print(df_to_latex(
    res.set_index(["country", "ID"]),
    roundings=[2, 2, 3, 2, 2, 3], hlines=False))

####### Compute DM tests between ID of the same OBs and same country
prices_pvalues, OB_pvalues = compute_DM_tests_OBs(
    countries,datasets,IDs,OB_sizes,predicted_prices, real_prices, predicted_OB,
    real_OB)

with matplotlib.rc_context({ "text.usetex" : True,
                             "text.latex.preamble" : r"\usepackage[bitstream-charter]{mathdesign} \usepackage[T1]{fontenc} \usepackage{mathtools}",
                             "font.family" : ""}):
    plt.close("all")    
    plot_DM_tests(
        prices_pvalues["20"], countries=countries, 
        IDs=IDs, label="prices")
    plt.show()

with matplotlib.rc_context({ "text.usetex" : True,
                             "text.latex.preamble" : r"\usepackage[bitstream-charter]{mathdesign} \usepackage[T1]{fontenc} \usepackage{mathtools}",
                             "font.family" : ""}):
    plt.close("all")    
    plot_DM_tests(
    OB_pvalues["20"], countries=countries, IDs=IDs, label="Order Books")
    plt.show()

###### COmpute DM tests between 2 OBs, fixed ID
price_tables = compute_DM_tests_2_OBs(
    countries,datasets,IDs, 20, 50, predicted_prices,
    real_prices, predicted_OB, real_OB)
print(df_to_latex(price_tables["FR"], hlines=False))
