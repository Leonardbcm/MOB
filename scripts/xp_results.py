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
XP_PARAMS = {
   "N_EPOCHS": 1000,
    "N_SAMPLES": 1444,
    "N_VAL": 365,
    "BATCH" : 80,
    "tboard" : "BETA_RESULTS",
}

CHECK_PARAMS = {
    # Shaps params
    "START": 0,
    "STOP" : 30,
    "STEP" : 30,
    "N_GROUPS" : 12,
    "N_SHAP" : 2,    

    # Other params
    "N_VAL": 365,
    "BATCH" : 80,   
    "tboard" : "CHECK",
}

PARAMS = XP_PARAMS

####### configurations
folder = PARAMS["tboard"]
#countries = ["FR", "DE", "BE", "NL"]
countries = ["BE"]
#datasets = ["Lyon", "Munich", "Bruges", "Lahaye"]
datasets = ["Bruges"]
IDs = np.array([1, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                23, 24, 25])
OB_sizes = np.array([20])
seeds = [0]

###### Retrieve several OBs
predicted_prices, real_prices, predicted_OB, real_OB, results=retrieve_results_OBs(
    IDs, countries, datasets, OB_sizes, PARAMS["N_VAL"], PARAMS["N_SAMPLES"],
    folder,nh = 24)
col_order = ["country","ID",
             "val_OB_smape", "val_OB_ACC","val_OB_rsmape", "val_OB_rACC",
             "val_price_mae", "val_price_dae", "val_price_rmae",
             "val_price_smape", "val_price_ACC"]
#col_order = ["seed", "val_OB_smape", "val_OB_ACC", "val_price_mae",
#            "val_price_smape", "val_price_ACC"]             
id_order = ["country", "ID"]
#id_order = ["seed"]
res = results["20"]
#res = results

res = res.sort_values(id_order).loc[:, col_order]
print(df_to_latex(
    res.set_index(id_order), roundings=[2, 3, 2, 2, 3, 2, 3], hlines=False))

##################### BETAS
plot_betas(res, np.array([1, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16]), "FR", "Lyon",
           ax_=None, col="val_price_smape")
plot_betas(res, np.array([3, 7, 17, 18, 19, 20, 21, 22]), "FR", "Lyon",
           ax_=None, col="val_price_smape")

plot_all_betas(res, np.array([1, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
               np.array([3, 7, 17, 18, 19, 20, 21, 22]), "FR", "Lyon")

IDs_to_use = np.array([1, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,23, 24, 25])
plot_all_betas_1(res, IDs_to_use, "BE", "Bruges", fontsize=20)

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

###### Compute DM tests between 2 OBs, fixed ID
price_tables = compute_DM_tests_2_OBs(
    countries,datasets,IDs, 20, 50, predicted_prices,
    real_prices, predicted_OB, real_OB)
print(df_to_latex(price_tables["FR"], hlines=False))

###### Shap Values
SHAP_PARAMS = {
    "N_EPOCHS": 1000,
    "N_SAMPLES": 1444,
    "N_GROUPS" : 12,    
    "START": 0,
    "STOP" : 365,
    "STEP" : 30,
    "N_SHAP" : 1000,        
    "N_VAL": 365,
    "BATCH" : 80,
    "tboard" : "BETA_RESULTS",
}
PARAMS = SHAP_PARAMS

IDs = [1, 8]
matrix = retrieve_shap_values(
    IDs, "BE", "Bruges", 20, PARAMS["N_VAL"], PARAMS["N_VAL"],
    PARAMS["N_GROUPS"], PARAMS["tboard"])

with matplotlib.rc_context({ "text.usetex" : True,
                             "text.latex.preamble" : r"\usepackage[bitstream-charter]{mathdesign} \usepackage[T1]{fontenc} \usepackage{mathtools}",
                             "font.family" : ""}):
    plt.close("all")
    params = {"fontsize_labels" : 15, "fontsize" : 20}
    plot_shap_values_beta(
        matrix,IDs,PARAMS["N_VAL"],"BE","Bruges",20,PARAMS["tboard"], params)
    plt.show()     
    
