%load aimport
import itertools, pandas, numpy as np, datetime, matplotlib.pyplot as plt, time
import matplotlib

from src.models.spliter import MySpliter
from src.models.torch_wrapper import OBNWrapper
from src.models.torch_models.weight_initializers import *
import src.models.parallel_scikit as ps
from src.analysis.utils import *
from src.analysis.xp_results_utils import *

%autoreload 1
%aimport src.analysis.xp_results_utils

"""
XP results and analysis file
"""

####### PARAMS
MAIN_PARAMS = {
    "N_EPOCHS": 1000,
    "N_SAMPLES": 1444,
    "N_VAL": 365,
    "BATCH" : 80,
    "tboard" : "RESULTS",
    "countries" : ["FR", "DE", "BE", "NL"],
    "datasets" : ["Lyon", "Munich", "Bruges", "Lahaye"],
    "IDs" : np.array([1, 2, 3, 4, 5, 6, 7]),
    "OB_sizes" : np.array([20]),
    "seeds" : [0],
}

BETA_PARAMS = {
    "N_EPOCHS": 1000,
    "N_SAMPLES": 1444,
    "N_VAL": 365,
    "BATCH" : 80,
    "tboard" : "BETA_RESULTS",
    "countries" : ["BE"],
    "datasets" : ["Bruges"],
    "IDs" : np.array([1, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                      21, 22, 23, 24, 25]),
    "OB_sizes" : np.array([20]),
    "seeds" : [0],
}

PREDICTION_PARAMS = {
    "N_EPOCHS": 1000,
    "N_SAMPLES": 1444,
    "N_VAL": 365,
    "BATCH" : 80,
    "tboard" : "BETA_RESULTS",
    "countries" : ["BE"],
    "datasets" : ["Bruges"],
    "IDs" : np.array([1, 6, 4]),
    "OB_sizes" : np.array([20]),
    "seeds" : [0],
}

OB_PARAMS = {
    "N_EPOCHS": 1000,
    "N_SAMPLES": 1444,
    "N_VAL": 365,
    "BATCH" : 80,
    "tboard" : "BETA_RESULTS",
    "countries" : ["BE"],
    "datasets" : ["Bruges"],
    "IDs" : np.array([4, 6]),
    "OB_sizes" : np.array([20]),
    "seeds" : [0],
}

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
    "countries" : ["BE"],
    "datasets" : ["Bruges"],
    "IDs" : np.array([1, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                      21, 22, 23, 24, 25]),
    "OB_sizes" : np.array([20]),
    "seeds" : [0],
    "tboard" : "SHAP_RESULTS",
}
###### Retrieve several OBs
PARAMS = MAIN_PARAMS
predicted_prices, real_prices, predicted_OB, real_OB, results=retrieve_results_OBs(
    PARAMS["IDs"], PARAMS["countries"], PARAMS["datasets"], PARAMS["OB_sizes"],
    PARAMS["N_VAL"], PARAMS["N_SAMPLES"],
    PARAMS["tboard"])
col_order = ["country","ID",
             "val_OB_smape", "val_OB_ACC","val_OB_rsmape", "val_OB_rACC",
             "val_price_mae", "val_price_dae", "val_price_rmae",
             "val_price_smape", "val_price_ACC"]
id_order = ["country", "ID"]
res = results["20"]

res = res.sort_values(id_order).loc[:, col_order]
print(df_to_latex(
    res.set_index(id_order), roundings=[2, 3, 2, 2, 3, 2, 3], hlines=False))

##################### DM TESTS
prices_pvalues, OB_pvalues = compute_DM_tests_OBs(
    PARAMS["countries"],PARAMS["datasets"],PARAMS["IDs"],PARAMS["OB_sizes"],
    predicted_prices, real_prices, predicted_OB, real_OB)

with matplotlib.rc_context({ "text.usetex" : True,
                             "text.latex.preamble" : r"\usepackage[bitstream-charter]{mathdesign} \usepackage[T1]{fontenc} \usepackage{mathtools}",
                             "font.family" : ""}):
    plt.close("all")
    params = {"fontsize" : 20, "fontsize_labels" : 15}
    plot_DM_tests(
        prices_pvalues["20"], params, countries=PARAMS["countries"], 
        IDs=PARAMS["IDs"], label="prices")
    plt.show()
##################### BETAS
PARAMS = BETA_PARAMS
predicted_prices, real_prices, predicted_OB, real_OB, results=retrieve_results_OBs(
    PARAMS["IDs"], PARAMS["countries"], PARAMS["datasets"], PARAMS["OB_sizes"],
    PARAMS["N_VAL"], PARAMS["N_SAMPLES"],
    PARAMS["tboard"])
with matplotlib.rc_context({ "text.usetex" : True,
                             "text.latex.preamble" : r"\usepackage[bitstream-charter]{mathdesign} \usepackage[T1]{fontenc} \usepackage{mathtools}",
                             "font.family" : ""}):
    plt.close("all")
    params = {"label_fontsize" : 15, "fontsize" : 20}    
    paper_plot_all_betas(results["20"], PARAMS["IDs"], 20, "BE", "Bruges", params)
    plt.show()
    
##################### Shap Values
PARAMS = SHAP_PARAMS
matrix = retrieve_shap_values(
    PARAMS["IDs"], "BE", "Bruges", 20, PARAMS["N_VAL"], PARAMS["N_VAL"],
    PARAMS["N_GROUPS"], PARAMS["tboard"])

with matplotlib.rc_context({ "text.usetex" : True,
                             "text.latex.preamble" : r"\usepackage[bitstream-charter]{mathdesign} \usepackage[T1]{fontenc} \usepackage{mathtools}",
                             "font.family" : ""}):
    plt.close("all")
    params = {"fontsize_labels" : 15, "fontsize" : 20}
    plot_shap_values_beta(
        matrix,PARAMS["IDs"],PARAMS["N_VAL"],"BE","Bruges",20,PARAMS["tboard"],
        params)
    plt.show()     
    
######## Plot prediction
PARAMS = PREDICTION_PARAMS
predicted_prices, real_prices, predicted_OB, real_OB, results=retrieve_results_OBs(
    PARAMS["IDs"], PARAMS["countries"], PARAMS["datasets"], PARAMS["OB_sizes"],
    PARAMS["N_VAL"], PARAMS["N_SAMPLES"],
    PARAMS["tboard"])
with matplotlib.rc_context({ "text.usetex" : True,
                             "text.latex.preamble" : r"\usepackage[bitstream-charter]{mathdesign} \usepackage[T1]{fontenc} \usepackage{mathtools}",
                             "font.family" : ""}):
    plt.close("all")
    params = {"fontsize_labels" : 20, "fontsize" : 25}
    plot_predictions(
        predicted_prices['20'][0], real_prices['20'][0], PARAMS["IDs"], 20,
        PARAMS["N_VAL"],"BE","Bruges", PARAMS["tboard"], params,
        betas_to_plot=[0,  0.5,  1])
    plt.show()

with matplotlib.rc_context({ "text.usetex" : True,
                             "text.latex.preamble" : r"\usepackage[bitstream-charter]{mathdesign} \usepackage[T1]{fontenc} \usepackage{mathtools}",
                             "font.family" : ""}):
    plt.close("all")
    params = {"fontsize_labels" : 20, "fontsize" : 35}
    plot_predictions(
        predicted_prices['20'][0], real_prices['20'][0], PARAMS["IDs"], 20,
        PARAMS["N_VAL"],"BE","Bruges", PARAMS["tboard"], params,
        betas_to_plot=[0,  0.5,  1], labels=["$DNN$", "$DO$", "$DNN + DO$"])
    plt.show()    

PARAMS = OB_PARAMS
dt = datetime.datetime(2019, 10, 12, 1)
OBs = predict_order_book(PARAMS["IDs"], 20, PARAMS["N_VAL"], "BE", "Bruges",
                         dt, PARAMS["tboard"])    
with matplotlib.rc_context({ "text.usetex" : True,
                             "text.latex.preamble" : r"\usepackage[bitstream-charter]{mathdesign} \usepackage[T1]{fontenc} \usepackage{mathtools}",
                             "font.family" : ""}):
    plt.close("all")
    params = {"fontsize_labels" : 15, "fontsize" : 20}
    plot_predicted_order_books(
        IDs, 20, PARAMS["N_VAL"],"BE","Bruges", dt, PARAMS["tboard"], params)
    plt.show()    

dataset = "Bruges"
country = "BE"
folder = PARAMS["tboard"]
N_VAL = PARAMS["N_VAL"]
OBs = 20
model_wrapper = OBNWrapper(
    "RESULTS", dataset, spliter=MySpliter(N_VAL, shuffle=False),country=country,
    skip_connection=True, use_order_books=False, order_book_size=OBs, IDn=4,
    tboard=folder)
X, Y = model_wrapper.load_train_dataset()
(Xtr, Ytr), (Xv, Yv) = model_wrapper.spliter(X, Y)
regr, version = model_wrapper.load(X, Y)

Xv_scaled = regr.steps[0][1].transform(Xv)
OBhat = regr.steps[1][1].predict_ob_scaled(Xv_scaled, regr)


######## Plot order book shrinking
SHRINK_PARAMS = {
    "country" : "BE",
    "dataset" : "Bruges",
    "OBs" : [20],
}

PARAMS = SHRINK_PARAMS

with matplotlib.rc_context({ "text.usetex" : True,
                             "text.latex.preamble" : r"\usepackage[bitstream-charter]{mathdesign} \usepackage[T1]{fontenc} \usepackage{mathtools}",
                             "font.family" : ""}):
    plt.close("all")
    params = {"fontsize_labels" : 25, "fontsize" : 30}
    plot_shrinking(SHRINK_PARAMS["country"], SHRINK_PARAMS["dataset"],
                   SHRINK_PARAMS["OBs"], dt, params)
    plt.show()


